"""
Generate 2-slide editable PPTX for ERDES dataset presentation.
Slide 1: Dataset overview with example images + class hierarchy
Slide 2: Task definitions with bottleneck analysis
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_CONNECTOR_TYPE, MSO_SHAPE
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(ROOT, "scripts", "example_frames")
CLIPS_DIR = os.path.join(ROOT, "scripts", "example_clips")
PIE_CHART = os.path.join(ROOT, "scripts", "dataset_stats_pie.png")
OUT_PPTX = os.path.join(ROOT, "scripts", "ERDES_Dataset_Intro.pptx")

# Colors
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
BLUE = RGBColor(0x2C, 0x3E, 0x50)
LIGHT_BLUE = RGBColor(0x34, 0x98, 0xDB)
GREEN = RGBColor(0x27, 0xAE, 0x60)
RED = RGBColor(0xE7, 0x4C, 0x3C)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
LIGHT_GRAY = RGBColor(0xEC, 0xF0, 0xF1)
ACCENT_BG = RGBColor(0xEB, 0xF5, 0xFB)
WARN_BG = RGBColor(0xFD, 0xED, 0xEC)

# Category info
CATEGORIES = [
    {"name": "Normal\n（正常眼）", "count": 4233, "pct": "78.7%", "color": GREEN,
     "file": "Normal.png", "video": "Normal.mp4", "urgency": "无需处理",
     "desc": "视网膜紧贴眼球后壁，呈光滑纤细的高回声线，无膜状回声"},
    {"name": "PVD\n（玻璃体后脱离）", "count": 646, "pct": "12.0%", "color": LIGHT_BLUE,
     "file": "PVD.png", "video": "PVD.mp4", "urgency": "定期随访",
     "desc": "纤细可移动膜状回声，不锚定于视神经，随眼球运动摆动明显"},
    {"name": "Macula Detached\n（黄斑脱离）", "count": 303, "pct": "5.6%", "color": RED,
     "file": "Macula_Detached.png", "video": "Macula_Detached.mp4", "urgency": "1-2周内择期手术",
     "desc": "明亮的V/Y形高回声膜，锚定于视盘，已延伸至黄斑区"},
    {"name": "Macula Intact\n（黄斑未脱离）", "count": 199, "pct": "3.7%", "color": ORANGE,
     "file": "Macula_Intact.png", "video": "Macula_Intact.mp4", "urgency": "24h内急诊手术",
     "desc": "明亮的V/Y形高回声膜，锚定于视盘，尚未延伸至黄斑区"},
]


def add_title(slide, text, left, top, width, height, font_size=24):
    """Add a styled title text box."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = True
    p.font.color.rgb = BLUE
    return txBox


def add_body_box(slide, text, left, top, width, height, font_size=11, bold=False, color=DARK_GRAY, align=PP_ALIGN.LEFT):
    """Add a body text box."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return txBox


def add_rounded_rect(slide, left, top, width, height, fill_color, text="", font_size=10, font_color=BLACK, bold=False):
    """Add a rounded rectangle shape with text."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = font_color
        p.alignment = PP_ALIGN.CENTER
        tf.paragraphs[0].space_before = Pt(2)
        tf.paragraphs[0].space_after = Pt(2)
    return shape


def add_image_safe(slide, path, left, top, width, height=None):
    """Add an image, skipping if not found."""
    if height is None:
        height = width
    if os.path.exists(path):
        return slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(width), Inches(height))
    else:
        # Placeholder rectangle
        shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)
        tf = shape.text_frame
        p = tf.paragraphs[0]
        p.text = "[Image not found]"
        p.font.size = Pt(9)
        p.font.color.rgb = DARK_GRAY
        p.alignment = PP_ALIGN.CENTER
        return shape


def build_slide1(prs):
    """Slide 1: Dataset Overview with example images and hierarchy."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

    # Title
    add_title(slide, "ERDES 数据集概览 —— 眼部B超视频分类", 0.5, 0.2, 9, 0.6, font_size=26)

    # Subtitle
    add_body_box(slide, "5,381 个视频片段 | 4 个诊断类别 | 5 个二分类任务\n"
                 "※ 数据集共覆盖 6 个去标识化采集对象（clip_id 中下划线前的唯一编号），每个 ID 对应一种亚型",
                 0.5, 0.7, 9, 0.5, font_size=10, color=RGBColor(0x7F, 0x8C, 0x8D))

    # ---- Image row: 4 example frames ----
    img_top = 1.15
    img_w = 1.85
    img_h = 1.85
    gap = 0.18
    start_x = 0.5

    for i, cat in enumerate(CATEGORIES):
        x = start_x + i * (img_w + gap)
        # Category label above
        add_body_box(slide, cat["name"].replace("\n", " "), x, img_top - 0.32, img_w, 0.3,
                     font_size=10, bold=True, color=cat["color"], align=PP_ALIGN.CENTER)
        # Video embed (fallback to static image if video not found)
        video_path = os.path.join(CLIPS_DIR, cat["video"])
        img_path = os.path.join(FRAMES_DIR, cat["file"])
        if os.path.exists(video_path):
            try:
                slide.shapes.add_movie(video_path, Inches(x), Inches(img_top), Inches(img_w), Inches(img_h),
                                       poster_frame_image=img_path if os.path.exists(img_path) else None)
            except Exception:
                # fallback to image
                add_image_safe(slide, img_path, x, img_top, img_w, img_h)
        else:
            add_image_safe(slide, img_path, x, img_top, img_w, img_h)
        # Count + urgency below image
        add_body_box(slide, f"{cat['count']} clips ({cat['pct']})  |  {cat['urgency']}",
                     x, img_top + img_h + 0.03, img_w, 0.25,
                     font_size=8, color=cat["color"], align=PP_ALIGN.CENTER, bold=True)
        # Feature description
        add_body_box(slide, cat["desc"],
                     x, img_top + img_h + 0.28, img_w, 0.35,
                     font_size=7.5, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    # ---- Classification hierarchy tree (left side) ----
    tree_top = 3.8
    tree_left = 0.5
    tree_w = 5.8

    # Title for tree section
    add_body_box(slide, "诊断分类层级", tree_left, tree_top, 3, 0.3, font_size=13, bold=True, color=BLUE)

    # Root node
    add_rounded_rect(slide, 1.8, tree_top + 0.35, 2.2, 0.35, BLUE, "全部数据 (5,381)", font_size=11, font_color=WHITE, bold=True)

    # Non-RD branch (left)
    add_rounded_rect(slide, 0.5, tree_top + 0.95, 1.5, 0.35, RGBColor(0xD5, 0xF5, 0xE3), "Non-RD (4,879)", font_size=10, bold=True)
    # RD branch (right)
    add_rounded_rect(slide, 2.3, tree_top + 0.95, 1.5, 0.35, RGBColor(0xFA, 0xDB, 0xD8), "RD (502)", font_size=10, bold=True)

    # Non-RD children
    add_rounded_rect(slide, 0.5, tree_top + 1.55, 1.5, 0.3, GREEN, "Normal: 4,233", font_size=9, font_color=WHITE)
    add_rounded_rect(slide, 0.5, tree_top + 1.95, 1.5, 0.3, LIGHT_BLUE, "PVD: 646", font_size=9, font_color=WHITE)

    # RD children
    add_rounded_rect(slide, 2.3, tree_top + 1.55, 1.5, 0.3, RED, "M. Detached: 303", font_size=9, font_color=WHITE)
    add_rounded_rect(slide, 2.3, tree_top + 1.95, 1.5, 0.3, ORANGE, "M. Intact: 199", font_size=9, font_color=WHITE)

    # ---- Pie chart (right side) ----
    add_image_safe(slide, PIE_CHART, 6.5, tree_top - 0.1, 3.0, 3.0)

    # ---- Key facts at bottom ----
    fact_top = 6.4
    add_rounded_rect(slide, 0.5, fact_top, 9.0, 0.9, ACCENT_BG,
                     "⭐  关键事实：极度类别不平衡 — RD 仅占 9.3%，但它是临床最需要检出的类别。\n"
                     "⭐  PVD 与 RD 在超声图像上极为相似（均呈膜状回声），是最容易混淆的类别对。\n"
                     "⭐  数据分布反映真实急诊场景：大多数就诊者无 RD，少数确诊 RD 中黄斑脱离占多数。",
                     font_size=10, font_color=DARK_GRAY, bold=False)
    # Left-align the fact text
    fact_shape = slide.shapes[-1]
    fact_shape.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT


def build_slide2(prs):
    """Slide 2: Task definitions and bottleneck analysis."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    add_title(slide, "五个分类任务 & 性能瓶颈分析", 0.5, 0.2, 9, 0.6, font_size=26)
    add_body_box(slide, "基于论文 ERDES 基准测试结果（3D U-Net / 3D ResNet）|  训练:验证:测试 = 72:8:20（分层采样）",
                 0.5, 0.65, 9, 0.35, font_size=11, color=RGBColor(0x7F, 0x8C, 0x8D))

    # ---- Main tasks table ----
    table_top = 1.2
    rows = 8  # header + 5 tasks + 2 extra rows
    cols = 6
    tbl_left = 0.4
    tbl = slide.shapes.add_table(rows, cols, Inches(tbl_left), Inches(table_top), Inches(9.2), Inches(3.3)).table

    # Column widths
    col_widths = [1.5, 1.3, 1.2, 0.7, 2.0, 2.5]
    for i, w in enumerate(col_widths):
        tbl.columns[i].width = Inches(w)

    # Table header
    headers = ["任务", "正类 (1)", "负类 (0)", "样本量", "难点分析", "基线 Sensitivity"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE

    # Table data
    tasks_data = [
        ["non_rd_vs_rd", "RD", "Non-RD\n(Normal+PVD)", "5,384", "极度不平衡 (RD仅9.3%)\n正类样本极少，易过拟合", "0.939  (3D ResNet)"],
        ["macula_detached\n_vs_intact", "Macula Intact\n(黄斑未脱离)", "Macula Detached\n(黄斑脱离)", "505", "样本量最少 (仅502 RD)\nRD内部细粒度区分", "0.899  (3D U-Net)"],
        ["normal_vs_pvd  ★", "PVD", "Normal", "4,879", "★ 最难任务！\nPVD回声弱、帧间可见性低\n需时序选择性池化策略", "0.784  (3D U-Net)"],
        ["normal_vs_rd", "RD", "Normal", "4,738", "PVD被排除，信号纯净\n不平衡 (RD仅10.6%)", "0.950  (3D U-Net)"],
        ["pvd_vs_rd", "RD", "PVD", "1,151", "临床最重要鉴别诊断\nPVD/RD超声表现高度相似", "0.921  (UNet++)"],
    ]

    for i, row_data in enumerate(tasks_data):
        for j, val in enumerate(row_data):
            cell = tbl.cell(i + 1, j)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9)
                p.font.color.rgb = DARK_GRAY
                if j == 0:
                    p.font.bold = True
                if j == 5:
                    p.font.bold = True
                    p.font.color.rgb = BLUE
                p.alignment = PP_ALIGN.CENTER if j >= 2 else PP_ALIGN.LEFT
            # Highlight the hardest task row
            if i == 2:  # normal_vs_pvd
                cell.fill.solid()
                cell.fill.fore_color.rgb = WARN_BG
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else LIGHT_GRAY
        # Set vertical alignment
        for j in range(cols):
            tbl.cell(i + 1, j).vertical_anchor = MSO_ANCHOR.MIDDLE

    # ---- Pipeline diagram at bottom ----
    pipe_top = 4.7
    add_body_box(slide, "诊断管线 (Diagnostic Pipeline)", 0.5, pipe_top, 4, 0.3,
                 font_size=13, bold=True, color=BLUE)

    # Pipeline boxes
    pipe_y = pipe_top + 0.4
    # Input
    add_rounded_rect(slide, 0.5, pipe_y, 1.5, 0.5, LIGHT_GRAY, "输入 B超视频", font_size=10, bold=True)
    # Stage 1
    add_rounded_rect(slide, 2.3, pipe_y, 2.2, 0.5, RGBColor(0xD4, 0xEF, 0xDF),
                     "Stage 1: non_rd_vs_rd\nSensitivity 0.939", font_size=9, bold=True)
    # If non-RD
    add_rounded_rect(slide, 2.5, pipe_y + 0.7, 1.8, 0.35, RGBColor(0xA9, 0xDF, 0xBF),
                     "→ Non-RD: 报告无RD", font_size=9)
    # Stage 2
    add_rounded_rect(slide, 4.8, pipe_y, 2.2, 0.5, RGBColor(0xFA, 0xDB, 0xD8),
                     "Stage 2: macula_detached\n_vs_intact  Sens 0.899", font_size=9, bold=True)
    # Output boxes
    add_rounded_rect(slide, 5.0, pipe_y + 0.7, 1.8, 0.35, RGBColor(0xF5, 0xB7, 0xB1),
                     "→ Macula Detached: 择期手术", font_size=8)
    add_rounded_rect(slide, 5.0, pipe_y + 1.15, 1.8, 0.35, RGBColor(0xFA, 0xD7, 0xA0),
                     "→ Macula Intact: 急诊手术", font_size=8)

    # E2E metric
    add_rounded_rect(slide, 0.5, pipe_y + 1.5, 3.6, 0.35, ACCENT_BG,
                     "端到端 Sensitivity: 0.844 (= 0.939 × 0.899)", font_size=11, font_color=BLUE, bold=True)

    # ---- Bottom insight ----
    insight_top = 6.65
    add_rounded_rect(slide, 0.5, insight_top, 9.0, 0.55, WARN_BG,
                     "💡  核心瓶颈：Normal vs PVD 是所有任务中最难的（Sensitivity 0.784 vs 其他任务 ≥0.899）。"
                     "原因：PVD 膜回声弱、仅部分帧可见，全局平均池化会稀释诊断信号。"
                     "改进方向：时序选择性池化（Selective Temporal Pooling）——在 96 帧中仅保留得分最高的 30%~70% 帧进行分类。",
                     font_size=10, font_color=DARK_GRAY, bold=False)
    slide.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    build_slide1(prs)
    build_slide2(prs)

    prs.save(OUT_PPTX)
    print(f"Saved: {OUT_PPTX}")


if __name__ == "__main__":
    main()
