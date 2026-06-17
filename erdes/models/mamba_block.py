"""
Pure PyTorch S6 Selective Scan Mamba Block
===========================================
Implements the core Mamba S6 mechanism without C++/CUDA compilation.
Based on: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces" (2023).

Works on Windows with pure Python — no mamba_ssm library required.

Core recurrence (S6):
    h_t = A_discrete(t) ⊙ h_{t-1} + B(t) ⊙ x_t
    y_t = C(t)^T · h_t + D · x_t

Shape convention:
    B = batch, L = sequence length, D = model dim, N = state dim
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScan(nn.Module):
    """
    Pure-PyTorch selective scan (S6 core).
    Iterates sequentially over L — correct but slower than CUDA kernel.

    Args:
        d_model:  model dimension (D)
        d_state:  state dimension (N), default 16
        dt_rank:  rank of Δ projection
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        # Δ projection: x → Δ (time-varying step size)
        self.dt_proj = nn.Linear(d_model, self.dt_rank)
        self.dt_scale = nn.Linear(self.dt_rank, d_model)

        # Input-dependent B and C projections
        # x_proj produces (B, C, Δ) combined
        self.x_proj = nn.Linear(d_model, d_state * 2 + self.dt_rank)

        # Learned A (diagonal state matrix, log-space for stability)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # D (skip connection, per-channel)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) — input sequence
        Returns:
            y: (B, L, D) — output sequence
        """
        B, L, D = x.shape

        # Project to produce B, C, Δ
        x_proj = self.x_proj(x)  # (B, L, 2*N + dt_rank)
        delta_rank = x_proj[..., :self.dt_rank]      # (B, L, dt_rank)
        B_in = x_proj[..., self.dt_rank:self.dt_rank + self.d_state]  # (B, L, N)
        C_in = x_proj[..., self.dt_rank + self.d_state:]               # (B, L, N)

        # Discretization: Δ = softplus(Linear(Linear(x)))
        delta = F.softplus(self.dt_scale(delta_rank))  # (B, L, D)

        # A ∈ R^{D×N} → discrete: A_d = exp(Δ ⊗ A)
        A = -torch.exp(self.A_log)  # (D, N), negative for stability
        A_d = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)

        # B_d = Δ · B_in  (Euler discretization)
        B_d = delta.unsqueeze(-1) * B_in.unsqueeze(2)  # (B, L, D, N)

        # Normalize C for stability
        C_norm = C_in / (C_in.norm(dim=-1, keepdim=True) + 1e-8)  # (B, L, N)

        # ---------- Parallel scan (O(log L) vs O(L)) ----------
        # Recurrence: h_t = a_t * h_{t-1} + b_t
        #   a_t = A_d[:,t]   shape (B, D, N)
        #   b_t = B_d[:,t] ⊙ x[:,t]  with broadcast over N
        b_t = B_d * x.unsqueeze(-1)  # (B, L, D, N)

        # Parallel prefix scan — non-inplace to preserve autograd graph
        a_p = A_d  # (B, L, D, N) — part of autograd graph, don't modify inplace
        h_p = b_t  # (B, L, D, N)
        step = 1
        while step < L:
            # For t >= step: combine (a_t, h_t) with (a_{t-step}, h_{t-step})
            # op(a2,b2) • op(a1,b1) = (a2*a1, a2*b1 + b2)
            a_comb = a_p[:, step:] * a_p[:, :-step]          # (B, L-step, D, N)
            h_comb = a_p[:, step:] * h_p[:, :-step] + h_p[:, step:]  # (B, L-step, D, N)
            # Non-inplace: cat prefix with combined suffix
            a_p = torch.cat([a_p[:, :step], a_comb], dim=1)
            h_p = torch.cat([h_p[:, :step], h_comb], dim=1)
            step *= 2

        # h_p now contains the hidden states
        # y_t = C_t · h_t  (inner product over N)
        y = (h_p * C_norm.unsqueeze(2)).sum(dim=-1)  # (B, L, D)

        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y


class MambaBlock(nn.Module):
    """
    Full Mamba block: LayerNorm → SelectiveScan with residual + gating.

    Args:
        d_model:  model dimension
        d_state:  SSM state dimension
        d_conv:   conv1d kernel size for local mixing
        expand:   expansion factor for inner dimension
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        inner_dim = d_model * expand

        self.norm = nn.LayerNorm(d_model)

        # Input projections
        self.in_proj = nn.Linear(d_model, inner_dim * 2)

        # 1D conv for local context
        self.conv1d = nn.Conv1d(
            in_channels=inner_dim, out_channels=inner_dim,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=inner_dim,  # depthwise
        )

        # Selective scan on the convolved features
        self.ssm = SelectiveScan(inner_dim, d_state)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            out: (B, L, D) with residual
        """
        residual = x
        x = self.norm(x)

        # Project input → (z, x) both of shape (B, L, inner_dim)
        proj = self.in_proj(x)  # (B, L, 2 * inner_dim)
        z, x_proj = proj.chunk(2, dim=-1)  # each (B, L, inner_dim)

        # 1D conv: (B, L, C) → (B, C, L) → conv → (B, C, L) → (B, L, C)
        x_conv = x_proj.transpose(1, 2)  # (B, inner_dim, L)
        L_seq = x_conv.shape[-1]           # Original sequence length
        x_conv = self.conv1d(x_conv)       # (B, inner_dim, L + padding)
        x_conv = x_conv[..., :L_seq]       # trim padding → (B, inner_dim, L)
        x_conv = F.silu(x_conv)
        x_conv = x_conv.transpose(1, 2)    # (B, L, inner_dim)

        # Selective scan
        y = self.ssm(x_conv)  # (B, L, inner_dim)

        # Gating and output
        y = y * F.silu(z)
        out = self.out_proj(y)  # (B, L, D)

        return out + residual


class Mamba3DProcessor(nn.Module):
    """
    Processes 3D feature maps by applying Mamba along three spatial axes
    (Depth-first, Height-first, Width-first), each in forward + backward order.

    Input:  [B, C, D, H, W]  (e.g. [1, 768, 6, 8, 8])
    Output: [B, C, D, H, W]  (same shape, with residual)

    Uses 6 independent MambaBlock instances for the 3 directions × 2 orders.
    Each direction flattens the 3D volume into a 1D sequence of ~384 tokens.
    """

    def __init__(self, channels: int = 768, d_state: int = 16,
                 d_conv: int = 4, expand: int = 1):
        super().__init__()
        C = channels

        # 3 directions × (forward + backward) = 6 Mamba blocks
        # Direction 1: HWL (Height → Width → Depth)
        self.mamba_hwl_f = MambaBlock(C, d_state, d_conv, expand)
        self.mamba_hwl_r = MambaBlock(C, d_state, d_conv, expand)

        # Direction 2: LWH (Depth → Width → Height)
        self.mamba_lwh_f = MambaBlock(C, d_state, d_conv, expand)
        self.mamba_lwh_r = MambaBlock(C, d_state, d_conv, expand)

        # Direction 3: LHW (Depth → Height → Width) — canonical order
        self.mamba_lhw_f = MambaBlock(C, d_state, d_conv, expand)
        self.mamba_lhw_r = MambaBlock(C, d_state, d_conv, expand)

        # Fusion: combine outputs from 3 directions
        # Each direction produces (fwd + rev)/2, so 3 directions total
        self.fusion = nn.Conv3d(C * 3, C, kernel_size=1)
        self.norm = nn.GroupNorm(8, C)

    @staticmethod
    def _process_direction(x_3d, mamba_f, mamba_r, permute_order, shape_3d):
        """
        Flatten 3D tensor along specified axis order, process with Mamba
        (forward + backward), reshape back to 3D.

        Args:
            x_3d:       (B, C, D, H, W)
            mamba_f:    forward MambaBlock
            mamba_r:    backward MambaBlock
            permute_order: how to permute dims before flatten
            shape_3d:   (B, C, D, H, W) — reference shape

        Returns:
            out: (B, C, D, H, W)
        """
        B, C, D, H, W = shape_3d

        # Permute to desired scan order, then flatten spatial dims
        x_p = x_3d.permute(*permute_order).contiguous()  # e.g. (B, C, H, W, D)
        L = x_p.shape[2] * x_p.shape[3] * x_p.shape[4]  # total tokens
        x_seq = x_p.reshape(B, C, L).transpose(1, 2)     # (B, L, C)

        # Forward scan
        out_f = mamba_f(x_seq)  # (B, L, C)

        # Backward scan (reverse sequence, process, reverse back)
        out_r = mamba_r(torch.flip(x_seq, dims=[1]))
        out_r = torch.flip(out_r, dims=[1])

        # Average forward + backward
        out = (out_f + out_r) / 2  # (B, L, C)

        # Reshape back: (B, L, C) → (B, C, L) → 3D → inverse permute
        out = out.transpose(1, 2).reshape(B, C, x_p.shape[2], x_p.shape[3], x_p.shape[4])

        # Inverse permute to restore (B, C, D, H, W)
        inv_order = [0] * 5
        for i, p in enumerate(permute_order):
            inv_order[p] = i
        out = out.permute(*inv_order).contiguous()

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            out: (B, C, D, H, W) with residual connection
        """
        B, C, D, H, W = x.shape
        shape_3d = (B, C, D, H, W)

        # Direction 1: scan along H→W→D axis
        o1 = self._process_direction(
            x, self.mamba_hwl_f, self.mamba_hwl_r,
            permute_order=(0, 1, 3, 4, 2),  # (B, C, H, W, D)
            shape_3d=shape_3d,
        )

        # Direction 2: scan along D→W→H axis
        o2 = self._process_direction(
            x, self.mamba_lwh_f, self.mamba_lwh_r,
            permute_order=(0, 1, 2, 4, 3),  # (B, C, D, W, H)
            shape_3d=shape_3d,
        )

        # Direction 3: scan along D→H→W axis (canonical)
        o3 = self._process_direction(
            x, self.mamba_lhw_f, self.mamba_lhw_r,
            permute_order=(0, 1, 2, 3, 4),  # (B, C, D, H, W) — unchanged
            shape_3d=shape_3d,
        )

        # Concatenate and fuse
        out = torch.cat([o1, o2, o3], dim=1)  # (B, 3C, D, H, W)
        out = self.fusion(out)                 # (B, C, D, H, W)
        out = self.norm(out)

        return out + x  # Residual connection


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing MambaBlock...")
    m = MambaBlock(d_model=128, d_state=16).cuda()
    x = torch.randn(2, 64, 128).cuda()
    y = m(x)
    print(f"  MambaBlock: {list(x.shape)} -> {list(y.shape)} OK")
    params = sum(p.numel() for p in m.parameters())
    print(f"  Parameters: {params:,}")
    print("  MambaBlock test: OK")

    print("\nTesting Mamba3DProcessor...")
    mp = Mamba3DProcessor(channels=768, d_state=16).cuda()
    x3d = torch.randn(1, 768, 6, 8, 8).cuda()
    y3d = mp(x3d)
    print(f"  Mamba3DProcessor: {list(x3d.shape)} -> {list(y3d.shape)} OK")
    params = sum(p.numel() for p in mp.parameters())
    print(f"  Parameters: {params:,}")
    print("All tests passed!")
