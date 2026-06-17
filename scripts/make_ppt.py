from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import math

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ═══════════════════════════════════════════
# Unified font sizes
# ═══════════════════════════════════════════
PAGE_TITLE  = 28   # section header Chinese title
PAGE_SUB_EN = 13   # section header English subtitle
H2          = 18   # content sub-heading (e.g. "核心痛点")
BODY        = 14   # body text
BODY_SMALL  = 12   # small body / notes
TABLE_FONT  = 11   # table cell text
SLIDE_NUM   = 9    # slide number
COVER_TITLE = 40   # cover page main title
COVER_DATE  = 16   # cover page date
TOC_NUM     = 16   # TOC item number
TOC_TITLE   = 18   # TOC item Chinese title
TOC_SUB     = 11   # TOC item English subtitle

FONT = "Microsoft YaHei"

# ═══════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════
BLACK       = RGBColor(0x1A, 0x1A, 0x1A)
DARK        = RGBColor(0x33, 0x33, 0x33)
GRAY        = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY  = RGBColor(0x99, 0x99, 0x99)
ACCENT      = RGBColor(0x1A, 0x56, 0x9A)
ACCENT_MID  = RGBColor(0x2B, 0x73, 0xC2)
ACCENT_LIGHT= RGBColor(0xDD, 0xE8, 0xF7)
ACCENT_PALE = RGBColor(0xF0, 0xF5, 0xFB)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
DARK_BG     = RGBColor(0x0D, 0x2B, 0x4E)
MID_BG      = RGBColor(0x14, 0x3D, 0x6E)
LIGHTER_BG  = RGBColor(0x1B, 0x4F, 0x8A)
BORDER      = RGBColor(0xDD, 0xDD, 0xDD)
ORANGE      = RGBColor(0xCC, 0x77, 0x00)

# ═══════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════

def set_slide_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_line(slide, left, top, width, color=BORDER, height=Pt(1.5)):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_rect(slide, left, top, width, height, color=ACCENT_LIGHT):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_textbox(slide, left, top, width, height, text="", font_size=BODY,
                bold=False, color=BLACK, alignment=PP_ALIGN.LEFT, font_name=FONT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_multi_text(slide, left, top, width, height, lines, font_name=FONT):
    """lines: list of (text, font_size, bold, color)"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(lines):
        text, fs, b, c = item[:4]
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(fs)
        p.font.bold = b
        p.font.color.rgb = c
        p.font.name = font_name
        p.space_after = Pt(6)
    return txBox

def add_slide_number(slide, num, total=10):
    add_textbox(slide, Inches(12.0), Inches(7.12), Inches(1.1), Inches(0.28),
                f"{num} / {total}", font_size=SLIDE_NUM, color=LIGHT_GRAY,
                alignment=PP_ALIGN.RIGHT)

def add_section_header(slide, title, subtitle=None):
    """Unified page header: accent bar + title + optional English subtitle + separator line"""
    add_rect(slide, Inches(0.7), Inches(0.55), Inches(0.06), Inches(0.48), ACCENT)
    add_textbox(slide, Inches(0.95), Inches(0.48), Inches(10), Inches(0.52),
                title, font_size=PAGE_TITLE, bold=True, color=BLACK)
    if subtitle:
        add_textbox(slide, Inches(0.95), Inches(0.95), Inches(10), Inches(0.32),
                    subtitle, font_size=PAGE_SUB_EN, color=GRAY)
    add_line(slide, Inches(0.7), Inches(1.32), Inches(11.9))

def add_h2(slide, left, top, text):
    """Unified sub-heading"""
    add_textbox(slide, left, top, Inches(5.5), Inches(0.35),
                text, font_size=H2, bold=True, color=ACCENT)

def draw_circle(slide, cx, cy, r, fill_color=None, line_color=None, line_width=Pt(1)):
    """Draw a circle centered at (cx, cy) with radius r"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, cx - r, cy - r, r * 2, r * 2)
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape

def draw_arc(slide, left, top, width, height, fill_color=None, line_color=None, line_width=Pt(0.5)):
    """Draw an arc shape"""
    shape = slide.shapes.add_shape(MSO_SHAPE.ARC, left, top, width, height)
    shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    return shape

# ═══════════════════════════════════════════
# SLIDE 1 — Title (with medical-themed BG)
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])

# Background gradient (simulated with many thin rects)
SW = prs.slide_width
SH = prs.slide_height
for i in range(60):
    frac = i / 60.0
    r = int(0x0D + frac * (0x1A - 0x0D))
    g = int(0x2B + frac * (0x56 - 0x2B))
    b = int(0x4E + frac * (0x9A - 0x4E))
    add_rect(slide, Inches(0), Emu(int(SW * i / 60)), SW, Emu(int(SH / 60 + 1)),
             RGBColor(r, g, b))

# Abstract wave/eye pattern — concentric faded circles (bottom-right)
cx = Inches(11.5); cy = Inches(3.0)
for amp, lc in [
    (0.65, RGBColor(0x3A, 0x6A, 0x9A)),
    (0.80, RGBColor(0x2E, 0x5E, 0x8E)),
    (1.00, RGBColor(0x24, 0x52, 0x80)),
    (1.25, RGBColor(0x1C, 0x46, 0x72)),
]:
    rx = int(Inches(amp) * 1.2)
    ry = int(Inches(amp))
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, cx - rx, cy - ry, rx * 2, ry * 2)
    shape.fill.background()
    shape.line.color.rgb = lc
    shape.line.width = Pt(1.5)

# Small accent dots
for dx, dy, r in [
    (Inches(2.0), Inches(5.0), Inches(0.25)),
    (Inches(2.5), Inches(5.45), Inches(0.18)),
    (Inches(9.5), Inches(1.5), Inches(0.3)),
    (Inches(10.2), Inches(1.85), Inches(0.2)),
    (Inches(3.0), Inches(1.8), Inches(0.22)),
]:
    draw_circle(slide, dx, dy, r, line_color=RGBColor(0x3A, 0x6A, 0x9A), line_width=Pt(0.8))

# Main title — editable textbox
add_textbox(slide, Inches(1.5), Inches(1.8), Inches(10.3), Inches(1.2),
            "基于时空特征的视网膜脱离\n超声视频智能诊断",
            font_size=COVER_TITLE, bold=True, color=WHITE,
            alignment=PP_ALIGN.CENTER)

# Separator
add_line(slide, Inches(5.5), Inches(3.3), Inches(2.3), WHITE, Pt(2.5))

# Subtitle — editable
add_textbox(slide, Inches(2), Inches(3.6), Inches(9.3), Inches(0.5),
            "ERDES: Eye Retinal Detachment Ultrasound — Benchmark & Diagnostic Pipeline",
            font_size=COVER_DATE, color=RGBColor(0xBB, 0xCC, 0xDD),
            alignment=PP_ALIGN.CENTER)

# Type line — editable
add_textbox(slide, Inches(2), Inches(4.8), Inches(9.3), Inches(0.45),
            "答辩汇报",
            font_size=22, bold=False, color=WHITE,
            alignment=PP_ALIGN.CENTER)

# Date — editable
add_textbox(slide, Inches(2), Inches(5.4), Inches(9.3), Inches(0.4),
            "2026年5月",
            font_size=COVER_DATE, color=RGBColor(0xBB, 0xCC, 0xDD),
            alignment=PP_ALIGN.CENTER)

add_slide_number(slide, 1)

# ═══════════════════════════════════════════
# SLIDE 2 — TOC (with themed BG)
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])

# Softer gradient background
for i in range(60):
    frac = i / 60.0
    r = int(0x15 + frac * (0x1F - 0x15))
    g = int(0x3D + frac * (0x55 - 0x3D))
    b = int(0x6E + frac * (0x85 - 0x6E))
    add_rect(slide, Inches(0), Emu(int(SW * i / 60)), SW, Emu(int(SH / 60 + 1)),
             RGBColor(r, g, b))

# Faded abstract circles (top-right)
for dx, dy, amp, alpha in [
    (Inches(11.2), Inches(1.0), Inches(1.5), 0.05),
    (Inches(11.2), Inches(1.0), Inches(1.0), 0.07),
]:
    rx = int(amp * 1.3)
    ry = int(amp)
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, dx - rx, dy - ry, rx * 2, ry * 2)
    shape.fill.background()
    shape.line.color.rgb = WHITE
    shape.line.width = Pt(0.8)

# Small dots
for dx, dy, r in [
    (Inches(1.5), Inches(5.5), Inches(0.12)),
    (Inches(2.0), Inches(5.8), Inches(0.08)),
    (Inches(11.5), Inches(6.0), Inches(0.15)),
]:
    draw_circle(slide, dx, dy, r, line_color=RGBColor(0x55, 0x77, 0x99), line_width=Pt(0.8))

# Title
add_textbox(slide, Inches(1.0), Inches(0.6), Inches(4), Inches(0.6),
            "目  录", font_size=PAGE_TITLE + 4, bold=True, color=WHITE)
add_textbox(slide, Inches(1.0), Inches(1.15), Inches(4), Inches(0.3),
            "Contents", font_size=PAGE_SUB_EN + 1, color=RGBColor(0xBB, 0xCC, 0xDD))
add_line(slide, Inches(0.95), Inches(1.5), Inches(11.3), RGBColor(0x55, 0x77, 0x99), Pt(1))

toc_items = [
    ("01", "研究背景与意义", "Background & Motivation"),
    ("02", "相关工作与技术演进", "Related Work"),
    ("03", "项目安排与进度", "Project Plan & Timeline"),
    ("04", "方法与实现 — 总体设计", "Methods: Two-Stage Pipeline"),
    ("05", "方法与实现 — 核心模块", "Methods: Key Modules"),
    ("06", "实验与结果分析", "Experiments & Analysis"),
    ("07", "改进方案", "Improvements"),
    ("08", "总结与展望", "Conclusion & Future Work"),
]

row_h = Inches(0.62)
left_col_x = Inches(1.2)
right_col_x = Inches(7.0)

for i, (num, zh, eng) in enumerate(toc_items):
    x = left_col_x if i < 4 else right_col_x
    y = Inches(1.85) + (i % 4) * row_h

    # Number
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y + Inches(0.03),
                                Inches(0.48), Inches(0.42))
    sp.fill.solid()
    sp.fill.fore_color.rgb = ACCENT_MID
    sp.line.fill.background()
    tf = sp.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.text = num
    p.font.size = Pt(TOC_NUM)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = FONT
    p.alignment = PP_ALIGN.CENTER

    add_textbox(slide, x + Inches(0.62), y - Inches(0.02), Inches(3.8), Inches(0.28),
                zh, font_size=TOC_TITLE, bold=True, color=WHITE)
    add_textbox(slide, x + Inches(0.62), y + Inches(0.28), Inches(3.8), Inches(0.22),
                eng, font_size=TOC_SUB, color=RGBColor(0xBB, 0xCC, 0xDD))
    add_line(slide, x + Inches(0.62), y + Inches(0.56), Inches(3.8),
             RGBColor(0x44, 0x66, 0x88), Pt(0.7))

add_slide_number(slide, 2)

# ═══════════════════════════════════════════
# Content slide template used for slides 3-10
# All content slides use WHITE background + add_section_header + add_h2 + BODY
# ═══════════════════════════════════════════

# ── Slide 3: 研究背景与意义 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, WHITE)
add_section_header(slide, "研究背景与意义", "Background & Motivation")

lx = Inches(0.95)
rx = Inches(7.2)

# -- left column --
add_h2(slide, lx, Inches(1.65), "核心痛点")
add_multi_text(slide, lx, Inches(2.1), Inches(5.5), Inches(1.5), [
    ("视网膜脱离（RD）是眼科急症，需紧急手术干预", BODY, False, DARK),
    ("黄斑状态（Macula-on/off）直接决定术后视力预后", BODY, False, DARK),
    ("超声是浑浊介质下唯一有效的眼底检查手段", BODY, False, DARK),
])

add_h2(slide, lx, Inches(3.5), "技术转型：从静态图像到超声视频")
add_multi_text(slide, lx, Inches(3.95), Inches(5.5), Inches(1.5), [
    ("利用时间维度冗余克服超声瞬时伪影", BODY, False, DARK),
    ("捕捉视网膜膜状物的动态波动特征", BODY, False, DARK),
    ("视频序列提供远超单帧的诊断信息量", BODY, False, DARK),
])

# -- right column --
sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, rx, Inches(1.65), Inches(5.1), Inches(2.3))
sp.fill.solid(); sp.fill.fore_color.rgb = ACCENT_PALE
sp.line.color.rgb = ACCENT; sp.line.width = Pt(1)

add_multi_text(slide, rx + Inches(0.4), Inches(1.85), Inches(4.3), Inches(1.8), [
    ("研究目标", H2, True, ACCENT),
    ("构建基于深度学习的超声视频自动诊断系统", BODY, False, DARK),
    ("实现两阶段级联分诊：RD检测 → 黄斑状态判定", BODY, False, DARK),
    ("在ERDES公开基准上达到SOTA性能", BODY, False, DARK),
])

sp2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, rx, Inches(4.15), Inches(5.1), Inches(1.8))
sp2.fill.solid(); sp2.fill.fore_color.rgb = ACCENT_PALE
sp2.line.color.rgb = ACCENT; sp2.line.width = Pt(1)

add_multi_text(slide, rx + Inches(0.4), Inches(4.35), Inches(4.3), Inches(1.4), [
    ("临床价值", H2, True, ACCENT),
    ("辅助基层医生快速筛查，降低漏诊率", BODY, False, DARK),
    ("为术前决策提供客观量化依据", BODY, False, DARK),
])

add_slide_number(slide, 3)

# ── Slide 4: 相关工作 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, WHITE)
add_section_header(slide, "相关工作与技术演进", "Related Work")

items = [
    ("CV 1.0", "手工特征时代", "SIFT、HOG等传统算子\n依赖专家经验设计\n泛化能力有限"),
    ("CV 2.0", "空间分割时代", "U-Net为核心的编解码架构\n跳跃连接实现多尺度融合\n医学图像分割新范式"),
    ("CV 3.0", "时空建模时代", "3D-CNN / Transformer\n同时捕获空间+时间特征\n视频理解能力大幅提升"),
]

for i, (label, title, desc) in enumerate(items):
    x = Inches(0.95 + i * 4.15)
    y = Inches(1.65)
    bw = Inches(3.75)
    bh = Inches(2.8)

    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, bw, bh)
    sp.fill.solid(); sp.fill.fore_color.rgb = ACCENT_PALE
    sp.line.color.rgb = BORDER; sp.line.width = Pt(1)

    add_textbox(slide, x + Inches(0.25), y + Inches(0.18), bw - Inches(0.5), Inches(0.35),
                label, font_size=22, bold=True, color=ACCENT)
    add_textbox(slide, x + Inches(0.25), y + Inches(0.58), bw - Inches(0.5), Inches(0.3),
                title, font_size=H2 - 2, bold=True, color=DARK)
    add_textbox(slide, x + Inches(0.25), y + Inches(1.1), bw - Inches(0.5), Inches(1.5),
                desc, font_size=BODY, color=DARK)

    if i < 2:
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + bw + Inches(0.05),
                                       y + Inches(1.2), Inches(0.35), Inches(0.3))
        arrow.fill.solid(); arrow.fill.fore_color.rgb = ACCENT
        arrow.line.fill.background()

add_h2(slide, Inches(0.95), Inches(4.85), "领域现状")
add_multi_text(slide, Inches(0.95), Inches(5.25), Inches(11.3), Inches(1.0), [
    ("该领域长期缺乏公开的大规模视频基准数据集，ERDES 是首个系统性解决此缺口的工作（5,381标注片段，8种架构基线）", BODY, False, DARK),
])

add_slide_number(slide, 4)

# ── Slide 5: 项目安排 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, WHITE)
add_section_header(slide, "项目安排与进度", "Project Plan & Timeline")

add_h2(slide, Inches(0.95), Inches(1.65), "任务分工")

tasks = [
    ("数据工程", "超声视频预处理与 ROI 自动提取算法实现"),
    ("模型研发", "8 种基线模型（3D ResNet, U-Net, Swin-UNETR等）的迁移与训练"),
    ("实验评估", "RD检测与黄斑状态判定的性能测试，多维度对比分析"),
]
for i, (role, desc) in enumerate(tasks):
    y = Inches(2.2 + i * 0.7)
    add_rect(slide, Inches(0.95), y + Inches(0.08), Inches(0.12), Inches(0.12), ACCENT)
    add_textbox(slide, Inches(1.25), y, Inches(2.2), Inches(0.35),
                role, font_size=BODY, bold=True, color=DARK)
    add_textbox(slide, Inches(3.0), y, Inches(7.5), Inches(0.35),
                desc, font_size=BODY, color=GRAY)

add_h2(slide, Inches(0.95), Inches(4.4), "进度安排")

stages = [
    ("第一阶段", "环境搭建与 ERDES 数据集解析", "已完成 ✓", ACCENT),
    ("第二阶段", "基线模型训练与两阶段管线跑通", "已完成 ✓", ACCENT),
    ("第三阶段", "结果对比分析与模型瓶颈挖掘", "进行中 ●", ORANGE),
    ("第四阶段", "针对性优化与论文撰写", "待开始 ○", LIGHT_GRAY),
]
for i, (stage, desc, status, st_color) in enumerate(stages):
    y = Inches(4.95 + i * 0.5)
    add_textbox(slide, Inches(1.25), y, Inches(2.0), Inches(0.32),
                stage, font_size=BODY, bold=True, color=DARK)
    add_textbox(slide, Inches(3.0), y, Inches(5.5), Inches(0.32),
                desc, font_size=BODY, color=DARK)
    add_textbox(slide, Inches(8.5), y, Inches(2.5), Inches(0.32),
                status, font_size=BODY_SMALL, bold=True, color=st_color)

add_slide_number(slide, 5)

# ── Slide 6: 方法 — 总体设计 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, WHITE)
add_section_header(slide, "方法与实现 — 总体设计", "Methods: Two-Stage Diagnostic Pipeline")

stages_data = [
    ("Stage 1: RD 检测", "Retinal Detachment Detection",
     ["输入：超声视频片段（D×H×W = 96×128×128）",
      "二分类：Non-RD vs RD（是否存在脱离）",
      "高灵敏度设计，优先降低假阴性",
      "使用所有训练数据训练"]),
    ("Stage 2: 黄斑状态判定", "Macula Status Classification",
     ["仅对 Stage 1 阳性样本执行",
      "二分类：Macula-intact vs Macula-detached",
      "针对 RD 阳性亚群精细化判定",
      "决定手术时机与预后评估"]),
]

for i, (title, eng, bullets) in enumerate(stages_data):
    x = Inches(0.95 + i * 6.15)
    y = Inches(1.65)

    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(5.75), Inches(3.0))
    sp.fill.solid(); sp.fill.fore_color.rgb = ACCENT_PALE
    sp.line.color.rgb = ACCENT; sp.line.width = Pt(1)

    add_textbox(slide, x + Inches(0.3), y + Inches(0.2), Inches(5.1), Inches(0.35),
                title, font_size=H2 + 2, bold=True, color=BLACK)
    add_textbox(slide, x + Inches(0.3), y + Inches(0.6), Inches(5.1), Inches(0.28),
                eng, font_size=PAGE_SUB_EN, color=GRAY)

    for j, b in enumerate(bullets):
        add_textbox(slide, x + Inches(0.5), y + Inches(1.15 + j * 0.42), Inches(5.0), Inches(0.32),
                    f"• {b}", font_size=BODY, color=DARK)

# Arrow
arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(6.72), Inches(2.7),
                               Inches(0.5), Inches(0.35))
arrow.fill.solid(); arrow.fill.fore_color.rgb = ACCENT
arrow.line.fill.background()
add_textbox(slide, Inches(6.0), Inches(3.1), Inches(1.95), Inches(0.28),
            "RD 阳性", font_size=BODY_SMALL, bold=True, color=ACCENT,
            alignment=PP_ALIGN.CENTER)

add_h2(slide, Inches(0.95), Inches(5.0), "设计理念")
add_multi_text(slide, Inches(0.95), Inches(5.4), Inches(11), Inches(1.0), [
    ("级联架构模拟临床分诊流程：先粗筛（有/无RD），再精判（黄斑状态）。Stage 2专注于RD阳性亚群，避免全量数据中类别不平衡带来的学习偏差。", BODY, False, DARK),
])

add_slide_number(slide, 6)

# ── Slide 7: 方法 — 核心模块 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, WHITE)
add_section_header(slide, "方法与实现 — 核心模块", "Methods: Key Modules")

# -- left column --
add_h2(slide, lx, Inches(1.65), "ROI 提取模块")
add_multi_text(slide, lx, Inches(2.1), Inches(5.5), Inches(1.0), [
    ("基于 YOLOv8 的眼球目标检测", BODY, False, DARK),
    ("自动裁剪眼球主体区域，消除设备文字等无效信息", BODY_SMALL, False, GRAY),
    ("减少无效背景计算，提升训练与推理效率", BODY_SMALL, False, GRAY),
])

add_h2(slide, lx, Inches(3.1), "特征提取骨干网络")

models_info = [
    ("卷积类", "3D U-Net · V-Net · ResNet3D · SENet154"),
    ("注意力类", "Vision Transformer (ViT) · Swin-UNETR · UNETR"),
    ("混合类", "UNet++（密集跳跃连接）"),
]
for i, (cat, models) in enumerate(models_info):
    y = Inches(3.55 + i * 0.5)
    add_textbox(slide, Inches(1.25), y, Inches(1.5), Inches(0.3),
                cat, font_size=BODY, bold=True, color=DARK)
    add_textbox(slide, Inches(2.6), y, Inches(5.5), Inches(0.3),
                models, font_size=BODY, color=GRAY)

add_h2(slide, lx, Inches(5.1), "选择性时间池化策略 (Selective Pooling)")
add_multi_text(slide, lx, Inches(5.55), Inches(5.5), Inches(1.2), [
    ("设定比例 r，将视频帧按特征重要性排序，仅保留 Top-r 关键帧", BODY, False, DARK),
    ("证明「全量视频帧 ≠ 最大信息量」，冗余帧反而引入噪声", BODY_SMALL, False, GRAY),
    ("在 Normal vs PVD 任务中，r=30% 时达到最优准确率 0.955", BODY_SMALL, False, GRAY),
])

# -- right column: flow diagram --
sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, rx, Inches(1.65), Inches(5.1), Inches(5.2))
sp.fill.solid(); sp.fill.fore_color.rgb = ACCENT_PALE
sp.line.color.rgb = ACCENT; sp.line.width = Pt(1)

add_textbox(slide, rx + Inches(0.3), Inches(1.8), Inches(4.5), Inches(0.3),
            "数据流与架构总览", font_size=H2, bold=True, color=ACCENT)

flow_items = [
    "① 输入视频 (D×H×W)",
    "↓",
    "② YOLOv8 ROI 检测与裁剪",
    "↓",
    "③ 3D Backbone 特征提取 (CNN / ViT / Hybrid)",
    "↓",
    "④ 选择性时间池化 (Top-r)",
    "↓",
    "⑤ FC → Sigmoid → 分类结果",
]

for i, item in enumerate(flow_items):
    is_arrow = item.startswith("↓")
    c = ACCENT if is_arrow else DARK
    b = item.startswith("①") or item.startswith("②") or item.startswith("③") or \
        item.startswith("④") or item.startswith("⑤")
    add_textbox(slide, rx + Inches(0.5), Inches(2.25) + Inches(i * 0.45),
                Inches(4.2), Inches(0.32),
                item, font_size=BODY, bold=b, color=c)

add_slide_number(slide, 7)

# ── Slide 8: 实验与分析 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, WHITE)
add_section_header(slide, "实验与结果分析", "Experiments & Analysis")

# -- left: setup + results table --
add_h2(slide, lx, Inches(1.65), "实验设置")
add_multi_text(slide, lx, Inches(2.1), Inches(5.5), Inches(1.5), [
    ("数据集：ERDES（5,381个超声视频片段）", BODY, False, DARK),
    ("输入尺寸：96 × 128 × 128 (D × H × W)", BODY, False, DARK),
    ("优化器：AdamW，学习率 1.5 × 10⁻⁵", BODY, False, DARK),
    ("评估指标：Accuracy / Sensitivity / Specificity", BODY, False, DARK),
    ("训练策略：5折交叉验证，固定随机种子保证可复现", BODY, False, DARK),
])

add_h2(slide, lx, Inches(3.8), "核心结果")

table_data = [
    ("任务", "最佳模型", "灵敏度", "特异度", "准确率"),
    ("Non-RD vs RD", "3D ResNet", "0.939", "0.978", "0.974"),
    ("Normal vs RD", "3D U-Net", "0.950", "0.996", "0.991"),
    ("PVD vs RD", "UNet++", "0.921", "0.860", "0.887"),
    ("Macula 判定", "3D U-Net", "0.899", "0.870", "0.882"),
    ("Normal vs PVD", "3D U-Net (r=30%)", "0.807", "0.977", "0.955"),
]

col_w = [Inches(1.6), Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.0)]
col_x = [Inches(0.95)]
for w in col_w[:-1]:
    col_x.append(col_x[-1] + w)

ty = Inches(4.25)
th = Inches(0.4)

for j, (hdr, cw, cx) in enumerate(zip(table_data[0], col_w, col_x)):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx, ty, cw, th)
    sp.fill.solid(); sp.fill.fore_color.rgb = ACCENT
    sp.line.color.rgb = ACCENT; sp.line.width = Pt(0.5)
    tf = sp.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = hdr
    p.font.size = Pt(TABLE_FONT); p.font.bold = True
    p.font.color.rgb = WHITE; p.font.name = FONT
    p.alignment = PP_ALIGN.CENTER

for i, row in enumerate(table_data[1:], 1):
    for j, (cell, cw, cx) in enumerate(zip(row, col_w, col_x)):
        bg = ACCENT_PALE if i % 2 == 0 else WHITE
        sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx, ty + i * th, cw, th)
        sp.fill.solid(); sp.fill.fore_color.rgb = bg
        sp.line.color.rgb = BORDER; sp.line.width = Pt(0.5)
        tf = sp.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = cell
        p.font.size = Pt(TABLE_FONT)
        p.font.bold = (j == 0)
        p.font.color.rgb = DARK; p.font.name = FONT
        p.alignment = PP_ALIGN.CENTER

# -- right: analysis --
add_h2(slide, rx, Inches(1.65), "结果分析")

add_multi_text(slide, rx, Inches(2.1), Inches(5.1), Inches(2.2), [
    ("瓶颈发现", H2, True, ACCENT),
    ("PVD（玻璃体后脱离）识别在各项中表现最弱", BODY, False, DARK),
    ("原因：PVD具有「高动态、低回声」特性，对瞬时特征捕捉能力要求更高", BODY_SMALL, False, GRAY),
    ("当前3D CNN在长程时序建模上存在局限", BODY_SMALL, False, GRAY),
])

add_multi_text(slide, rx, Inches(4.2), Inches(5.1), Inches(2.2), [
    ("采样策略的启示", H2, True, ACCENT),
    ("选择性采样比例 r 对性能影响显著", BODY, False, DARK),
    ("Normal vs PVD 中 unet3d 不同 r 的对比：", BODY_SMALL, False, GRAY),
    ("r=30% 准确率 0.955 > r=100% 准确率", BODY_SMALL, False, GRAY),
    ("→ 全量视频帧 ≠ 最大信息量", BODY, False, ACCENT),
])

add_slide_number(slide, 8)

# ── Slide 9: 改进方案 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, WHITE)
add_section_header(slide, "改进方案", "Improvements")

add_h2(slide, Inches(0.95), Inches(1.65), "当前模型的主要问题")

problems = [
    "PVD 识别性能瓶颈：模型对低回声条件下的动态特征捕捉能力不足",
    "固定采样策略的局限：选择性采样比例 r 需人工设定，无法自适应不同病例",
    "两阶段级联误差累积：Stage 1 的假阴性在 Stage 2 无法被纠正",
]
for i, p in enumerate(problems):
    add_textbox(slide, Inches(1.25), Inches(2.15 + i * 0.5), Inches(10.5), Inches(0.32),
                f"• {p}", font_size=BODY, color=DARK)

add_h2(slide, Inches(0.95), Inches(3.75), "改进方案")

improvements = [
    ("可学习的注意力采样", "Attention-based Adaptive Sampling",
     ["替换固定比例 r 为可学习的注意力权重",
      "在特征金字塔（FPN）后接入轻量注意力分支",
      "自适应选择具有病理信息的关键帧",
      "预期：提升PVD识别性能，减少固定采样的信息损失"]),
    ("强化时序建模能力", "Enhanced Temporal Modeling",
     ["探索 VideoMAE / TimeSformer 等Video Transformer",
      "研究时序自监督预训练在超声数据上的迁移",
      "在Swin-UNETR中引入时序偏移窗口",
      "预期：增强对低回声动态特征的捕获"]),
]

for i, (title, eng, bullets) in enumerate(improvements):
    x = Inches(0.95 + i * 6.15)
    y = Inches(4.3)

    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(5.75), Inches(2.7))
    sp.fill.solid(); sp.fill.fore_color.rgb = ACCENT_PALE
    sp.line.color.rgb = ACCENT; sp.line.width = Pt(1)

    add_textbox(slide, x + Inches(0.3), y + Inches(0.18), Inches(5.1), Inches(0.32),
                title, font_size=H2, bold=True, color=BLACK)
    add_textbox(slide, x + Inches(0.3), y + Inches(0.52), Inches(5.1), Inches(0.26),
                eng, font_size=BODY_SMALL, color=GRAY)

    for j, b in enumerate(bullets):
        add_textbox(slide, x + Inches(0.5), y + Inches(0.95 + j * 0.4), Inches(5.0), Inches(0.3),
                    f"• {b}", font_size=BODY, color=DARK)

add_slide_number(slide, 9)

# ── Slide 10: 结论 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, WHITE)
add_section_header(slide, "总结与展望", "Conclusion & Future Work")

add_h2(slide, Inches(0.95), Inches(1.65), "工作总结")

summary_items = [
    "成功复现 ERDES 基线实验，跑通完整的两阶段诊断管线",
    "在两阶段级联架构下验证了时空特征在RD诊断中的有效性",
    "系统对比了 8 种 3D 架构（CNN/ViT/混合）在 5 项分类任务上的性能",
    "分析并定位了 PVD 识别瓶颈，为后续改进指明了方向",
]
for i, item in enumerate(summary_items):
    add_textbox(slide, Inches(1.25), Inches(2.15 + i * 0.5), Inches(10.5), Inches(0.32),
                f"✓ {item}", font_size=BODY, color=DARK)

add_h2(slide, Inches(0.95), Inches(4.35), "未来方向")

sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.95), Inches(4.85), Inches(11.3), Inches(2.0))
sp.fill.solid(); sp.fill.fore_color.rgb = ACCENT_PALE
sp.line.color.rgb = ACCENT; sp.line.width = Pt(1)

add_multi_text(slide, Inches(1.3), Inches(5.0), Inches(10.6), Inches(1.6), [
    ("核心方向：可学习的注意力采样机制", H2, True, ACCENT),
    ("当前选择性采样中的比例 r 为人工固定值，下一步计划引入可学习的注意力采样模块，"
     "实现对诊断关键帧的自适应捕获。结合 Grad-CAM 可解释性分析验证注意力权重与"
     "临床病变区域的一致性，为模型决策提供更透明、可信的临床依据。", BODY, False, DARK),
])

add_slide_number(slide, 10)

# ── Save ──
out = r"D:\projects\ERDES\ERDES_答辩汇报_v2.pptx"
prs.save(out)
print(f"Saved to: {out}")
