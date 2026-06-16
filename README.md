# ERDES — Retinal Detachment Diagnosis from Ocular Ultrasound Videos

[![arXiv](https://img.shields.io/badge/arXiv-2508.04735-b31b1b.svg)](https://arxiv.org/abs/2508.04735)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

基于眼部超声视频的**两阶段级联分诊系统**：Stage 1 检测视网膜脱离（RD），Stage 2 判定黄斑状态（Macula Detached vs Intact）。复现并改进了 [ERDES 论文](https://arxiv.org/abs/2508.04735) 的 8 种 3D 基线模型，引入 **纯 PyTorch Mamba（S6 选择性状态空间模型）** 替代 CUDA 依赖，在瓶颈层实现三轴双向序列建模。

---

##  项目结构

```
ERDES/
├── erdes/
│   ├── train.py / eval.py          # Hydra驱动的训练/评估入口
│   └── models/
│       ├── components/
│       │   ├── cls_model.py          # 8种3D分类器 (UNet3D/ResNet/SwinUNETR/…)
│       │   ├── factory.py            # 架构工厂
│       │   └── encoders/             # 编码器 (unet3d/unet++/vnet/vit/…)
│       ├── mamba_block.py            # ★ 纯PyTorch S6选择性扫描
│       ├── hybrid_unet_mamba.py      # ★ CNN+Mamba混合架构
│       └── explainer.py              # 3D Grad-CAM可解释性
├── scripts/
│   ├── batch_eval_5tasks.py          # 5任务基线批量评估
│   ├── exp_mamba_ablation.py         # Mamba消融实验
│   ├── analysis_data_stats.py        # 运动分数统计分析
│   ├── visualize_results.py          # 全部可视化
│   └── …
├── results/                          # 所有实验结果
│   ├── mamba_ablation/               # Mamba改进结果 (F1+0.01)
│   ├── exp/                          # 消融 + 流水线
│   ├── stats/                        # 运动分数
│   └── plots/                        # 可视化
├── configs/                          # Hydra配置
├── data/splits/                      # 官方 train/val/test 划分
└── weights/                          # HuggingFace预训练权重
```

---

##  安装

```bash
conda create -n erdes python=3.10
conda activate erdes
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install lightning hydra-core monai safetensors pandas tqdm matplotlib decord
```

**硬件:** NVIDIA GPU (≥8 GB VRAM)。Mamba 模块纯 PyTorch 实现，Windows/Linux 兼容。

---

##   5 个分类任务

| 任务 | 描述 | 测试样本 | 论文最优模型 | 论文 Acc | 实测 Acc |
|:---|:---|:---:|:---|:---:|:---:|
| Non-RD vs. RD | RD 检测 | 1,077 | 3D ResNet + 3D U-Net | 0.974 / 0.982 | 0.9749 / 0.9824 |
| Normal vs. RD | 正常 vs 脱离 | 945 | 3D U-Net | 0.991 | 0.9937 |
| PVD vs. RD | 玻璃体脱离 vs 脱离 | 230 | UNet++ | 0.887 | 0.8870 |
| Macula Detached vs. Intact | 黄斑状态 | 101 | 3D U-Net | 0.882 | 0.8911 |
| Normal vs. PVD | 正常 vs 玻璃体脱离 | 976 | 3D U-Net | 0.959 | 0.9590 |

**全部基线偏差 < ±0.02，完整复现论文。**

---

##   数据预处理

```
原始视频 (.mp4, yuv420p)
  → decord / torchvision 解码        [T, H, W, C] uint8
  → .float().permute(3,0,1,2)         [C, D, H, W] float32
  → .mean(dim=0, keepdim=True)        灰度化 [1, D, H, W]
  → PadToSquare3D()                   零填充至正方形
  → Interpolate3D("trilinear", 96×128×128)
  → / 255.0                           归一化 [0, 1]
```

---

##   核心架构

### Baseline: 3D U-Net (25M params)

```
Input [1,96,128,128]
  → UNet3DEncoder (5层, GroupNorm, f_maps=[64,128,256,512,768])
     每层: MaxPool3d(k=2) + 2×Conv3d(3³)
  → Bottleneck [768, 6, 8, 8]
  → ClassificationHead: AdaptiveAvgPool3d → FC(768→384) → FC(384→1) → Sigmoid
```

### 改进: Hybrid UNet-Mamba (38M params)

```
Input [1,96,128,128]
  → UNet3DEncoder (预训练权重加载)
  → ★ Mamba3DProcessor (384 tokens, 3轴双向扫描)
       HWL (空间→时间) + LWH (时间→空间) + LHW (时间→高度)
       每方向: Mamba fwd + Mamba rev → 平均 → Concat → Conv3d融合 → 残差
  → ClassificationHead (复用预训练权重) → Sigmoid
```

**Mamba 加速:** 并行前缀扫描 O(log L) 替代串行 O(L)。384 tokens 仅需 9 步并行归约。

---

##   快速开始

```bash
# 1. 5任务基线评估 (预训练权重直接推理)
python scripts/batch_eval_5tasks.py

# 2. Mamba消融实验 (微调train + test)
python scripts/exp_mamba_ablation.py

# 3. 运动分数分析
python scripts/analysis_data_stats.py

# 4. 可视化
python scripts/visualize_results.py
python scripts/viz_mamba_final.py
```

---

##   核心结果

### Mamba 改进 (macula_detached_vs_intact)

| 指标 | Baseline UNet3D | Hybrid UNet-Mamba | Δ |
|:---|:---:|:---:|:---:|
| Accuracy | 0.8911 | **0.9010** | **+0.0099** |
| Sensitivity | 0.9000 | 0.9000 | — |
| Specificity | 0.8852 | **0.9016** | **+0.0164** |
| F1 Score | 0.8675 | **0.8780** | **+0.0105** |

训练配置: AdamW (lr=1e-5), 混合精度 FP16, 早停 (F1, patience=3)。epoch 7 达到峰值。

### 级联流水线

```
Stage 1 (RD 检测):      Sensitivity = 0.9200 (UNet3D)
Stage 2 (黄斑判定):     Sensitivity = 0.9000 (UNet3D)
Combined Sensitivity:  0.9200 × 0.9000 = 0.8280
```

### 采样消融 (macula, r ∈ {0.3, 0.5, 0.7, 1.0})

小型测试集上 4 个比例指标一致 (F1≈0.868)，采样比例对该任务影响不显著。

---

##   训练配置

| 配置项 | 值 |
|:---|:---|
| 优化器 | AdamW (lr=1×10⁻⁵, weight_decay=0.01) |
| 损失函数 | BCEWithLogitsLoss |
| Batch Size | 1 (RTX 5060 8GB) |
| 混合精度 | FP16 (`torch.cuda.amp`) |
| 早停 | F1 监控, patience=3, 排除退化解 |
| 数据增强 | 无（医疗影像谨慎策略） |

---

##   引用

```bibtex
@article{erdes2025,
  title   = {ERDES: A Benchmark Video Dataset for Retinal Detachment and
             Macular Status Classification in Ocular Ultrasound},
  author  = {Ozkut, Yasemin and Navard, Pouyan and Adhikari, Srikar and
             Situ-LaCasse, Elaine and Acuña, Josie and Yarnish, Adrienne A
             and Yilmaz, Alper},
  journal = {arXiv preprint arXiv:2508.04735},
  year    = {2025}
}
```

##   数据与权重

- **数据集:** [HuggingFace](https://huggingface.co/datasets/pcvlab/erdes) | [Zenodo](https://zenodo.org/records/18644370)
- **预训练权重:** [HuggingFace](https://huggingface.co/collections/pcvlab/erdes-ocular-ultrasound-classification-all-tasks) | [Zenodo](https://zenodo.org/records/18821031)
- **YOLOv8 眼球检测:** [HuggingFace](https://huggingface.co/pcvlab/yolov8_ocular_ultrasound_globe_detection)

## 致谢

本项目基于 [ERDES 论文](https://arxiv.org/abs/2508.04735) 和 [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template)。
