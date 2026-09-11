#!/usr/bin/env python3
"""
Generate Contribution Report PDF
=================================
Compiles the official team contribution report required per course project
specifications (§6). Details individual member ownership, technical
contributions, code modules, git commits, and percentage distribution.
Formatted to fit cleanly onto a single, high-density executive page.
"""

import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def create_contribution_report(output_pdf_path: str):
    # Page setup: letter is 8.5 x 11 inches (612 x 792 points)
    # Margins: 28pt left/right, 24pt top/bottom -> 556pt width x 744pt printable height
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=28,
        rightMargin=28,
        topMargin=24,
        bottomMargin=24,
    )

    styles = getSampleStyleSheet()

    # Color Palette
    primary_color = colors.HexColor("#1A365D")    # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0")  # Slate Blue
    accent_color = colors.HexColor("#2C5282")     # Dark Slate
    text_dark = colors.HexColor("#2D3748")        # Charcoal text
    bg_light = colors.HexColor("#F7FAFC")         # Off-white / light tint
    border_color = colors.HexColor("#CBD5E0")     # Subtle border
    highlight_bg = colors.HexColor("#EBF8FF")     # Soft blue highlight

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=primary_color,
        alignment=1,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=secondary_color,
        alignment=1,
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=text_dark,
        alignment=1,
    )

    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=primary_color,
        spaceBefore=4,
        spaceAfter=2,
    )

    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=text_dark,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.4,
        textColor=text_dark,
        leftIndent=8,
        firstLineIndent=-6,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=8.8,
        textColor=text_dark,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell_style,
        alignment=1,
    )

    story = []

    # 1. Header & Metadata
    story.append(Paragraph("TEAM CONTRIBUTION & WORK DISTRIBUTION REPORT", title_style))
    story.append(Spacer(1, 2))
    story.append(
        Paragraph(
            "Course: DLE-AI-202 (Deep Learning), Cohort I 2026 &nbsp;|&nbsp; Track 1: Pure Research",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 2))
    story.append(
        Paragraph(
            "<b>Project:</b> <i>Do Register Tokens Regularize Vision Transformers Under Data Scarcity? A Controlled 12-Run Ablation</i><br/>"
            "<b>Repository:</b> github.com/shahin1717/vit-research &nbsp;|&nbsp; <b>Submission Date:</b> September 8, 2026",
            meta_style,
        )
    )
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=1.2, color=primary_color, spaceAfter=4))

    # 2. Executive Overview (§6 Compliance)
    story.append(Paragraph("1. Executive Overview & Compliance Statement (§6)", h1_style))
    story.append(
        Paragraph(
            "Per Section 6 of the project specifications, this report documents the workload allocation, code ownership, "
            "and technical deliverables across our team. The project investigated the regularizing capacity of register tokens "
            "(K &isin; {0, 1, 4, 8}) under extreme data scarcity (~10k CIFAR-100 images). All five members contributed equally "
            "(<b>20.0% per member</b>) across theoretical formulation, PyTorch engineering, A100 GPU sweeps, statistical reduction, "
            "and manuscript drafting.",
            body_style,
        )
    )
    story.append(Spacer(1, 3))

    # 3. Summary Distribution Table
    story.append(Paragraph("2. Responsibility Matrix & Effort Allocation", h1_style))
    headers = [
        Paragraph("Member Name", table_header_style),
        Paragraph("GitHub / Contact", table_header_style),
        Paragraph("Core Ownership Domain", table_header_style),
        Paragraph("Primary Assigned Codebase Modules", table_header_style),
        Paragraph("Effort", table_header_style),
    ]

    rows = [
        headers,
        [
            Paragraph("<b>Shahin Alakparov</b>", table_cell_style),
            Paragraph("shahin1717<br/>sahinalekperov5@gmail.com", table_cell_style),
            Paragraph("Core Architecture &amp; Training Lead", table_cell_style),
            Paragraph("<code>src/models/register_vit.py</code>, <code>scripts/train.py</code>, <code>scripts/eval.py</code>", table_cell_style),
            Paragraph("<b>20.0%</b>", table_cell_center),
        ],
        [
            Paragraph("<b>Gulnisa Abdurahmanli</b>", table_cell_style),
            Paragraph("gulnisa<br/>gulnisa.abdurahmanli@gmail.com", table_cell_style),
            Paragraph("Data Engineering &amp; Multi-Budget Lead", table_cell_style),
            Paragraph("<code>src/data/cifar100_subset.py</code> (100 &amp; 300 imgs/class), <code>src/data/__init__.py</code>", table_cell_style),
            Paragraph("<b>20.0%</b>", table_cell_center),
        ],
        [
            Paragraph("<b>Narmina Ibrahimova</b>", table_cell_style),
            Paragraph("nnrmina<br/>narminaibrahimova2@gmail.com", table_cell_style),
            Paragraph("Metrics &amp; Interpretability Lead", table_cell_style),
            Paragraph("<code>src/models/attention_hook.py</code>, <code>src/metrics/entropy.py, outliers.py</code>", table_cell_style),
            Paragraph("<b>20.0%</b>", table_cell_center),
        ],
        [
            Paragraph("<b>Emil Ahmedli</b>", table_cell_style),
            Paragraph("emilahmedli5<br/>emilahmedli1905@gmail.com", table_cell_style),
            Paragraph("Ablation Sweeps &amp; Hardware Lead", table_cell_style),
            Paragraph("<code>scripts/run_sweep.sh</code>, <code>scripts/run_databudget_sweep.sh</code>, <code>configs/</code>", table_cell_style),
            Paragraph("<b>20.0%</b>", table_cell_center),
        ],
        [
            Paragraph("<b>Rufet Dosteliyev</b>", table_cell_style),
            Paragraph("rufetdosteliyev<br/>rufetdosteliyev@gmail.com", table_cell_style),
            Paragraph("Ablation Analytics, Figures &amp; Paper Lead", table_cell_style),
            Paragraph("<code>src/utils/aggregate_databudget.py</code>, <code>scripts/plot_*.py</code>, <code>paper/</code>", table_cell_style),
            Paragraph("<b>20.0%</b>", table_cell_center),
        ],
        [
            Paragraph("<b>TOTALS</b>", table_cell_style),
            Paragraph("<b>5 Team Members</b>", table_cell_style),
            Paragraph("<b>Balanced End-to-End Pure Research</b>", table_cell_style),
            Paragraph("<b>24/24 Ablation Runs Verified (100% Passing Tests)</b>", table_cell_style),
            Paragraph("<b>100.0%</b>", table_cell_center),
        ],
    ]

    col_widths = [1.25 * inch, 1.55 * inch, 1.65 * inch, 2.7 * inch, 0.55 * inch]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_color),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, bg_light]),
            ("BACKGROUND", (0, -1), (-1, -1), highlight_bg),
            ("LINEABOVE", (0, -1), (-1, -1), 1.0, secondary_color),
        ])
    )
    story.append(t)
    story.append(Spacer(1, 4))

    # 4. Granular Technical Deliverables
    story.append(Paragraph("3. Detailed Technical Contributions per Member", h1_style))

    story.append(Paragraph(
        "&bull; <b>Shahin Alakparov:</b> Engineered <code>RegisterVisionTransformer</code> wrapping <code>timm</code> ViT-Tiny (d=192). "
        "Implemented prepend logic for learnable registers R &isin; &real;<sup>K &times; d</sup> and output slicing. Developed the AMP training harness in "
        "<code>scripts/train.py</code> (AdamW, linear warmup + cosine annealing, gradient clipping, checkpoint serialization) and verified 59/59 tests.",
        bullet_style,
    ))
    story.append(Paragraph(
        "&bull; <b>Gulnisa Abdurahmanli:</b> Built generalized <code>StratifiedCIFAR100Subset</code> in <code>src/data/cifar100_subset.py</code>, sampling both "
        "100 and 300 samples/class (10k and 30k images) with deterministic generator seeds. Implemented balanced 9k/1k and 27k/3k train/val splits, Bicubic "
        "RandomResizedCrop(224), AutoAugment, and standard test set evaluation loader with pinned memory and multi-worker prefetching.",
        bullet_style,
    ))
    story.append(Paragraph(
        "&bull; <b>Narmina Ibrahimova:</b> Authored <code>ViTAttentionHookManager</code> in <code>src/models/attention_hook.py</code>, non-invasively "
        "intercepting attention weights across all 12 MHSA layers with automatic CUDA memory clearing. Coded Shannon attention entropy H(A<sup>(l)</sup>) in "
        "<code>src/metrics/entropy.py</code>, patch-norm 3&sigma; outlier rate in <code>src/metrics/outliers.py</code>, and generalization gap metrics.",
        bullet_style,
    ))
    story.append(Paragraph(
        "&bull; <b>Emil Ahmedli:</b> Led execution of all 24 ablation sweep runs across both data budgets (12 &times; 100pc and 12 &times; 300pc). "
        "Configured WireGuard VPN (<code>team1.conf</code>) and remote NVIDIA A100-SXM4 GPU cluster environment. Authored automated execution runners "
        "<code>scripts/run_sweep.sh</code> and <code>scripts/run_databudget_sweep.sh</code> with GPU cache clearing, supervising zero-OOM execution across 8+ hours.",
        bullet_style,
    ))
    story.append(Paragraph(
        "&bull; <b>Rufet Dosteliyev:</b> Led multi-seed ablation analytics, statistical reduction, and scientific reporting across both 100pc and 300pc "
        "experimental suites. Built aggregation engines (<code>src/utils/logger.py</code> and <code>src/utils/aggregate_databudget.py</code>) producing "
        "<code>sweep_summary.json</code> and <code>summary.json</code>. Authored comparative visualizers (<code>scripts/plot_metrics.py</code>, "
        "<code>scripts/plot_databudget_comparison.py</code>), LaTeX tables, and manuscript Section 5.",
        bullet_style,
    ))
    story.append(Spacer(1, 4))

    # 5. Attestation & Signatures
    story.append(Paragraph("4. Academic Integrity Attestation & Sign-Off", h1_style))
    story.append(
        Paragraph(
            "We certify that this report accurately reflects the collaborative division of labor for our final project. "
            "All members actively reviewed, validated, and approved the codebase, empirical findings, manuscript, and oral defense deck.",
            body_style,
        )
    )
    story.append(Spacer(1, 3))

    sig_data_1 = [
        [
            Paragraph("<b>Shahin Alakparov</b><br/><i>Core Architecture Lead</i><br/>Status: <b>Verified &amp; Signed</b>", table_cell_style),
            Paragraph("<b>Gulnisa Abdurahmanli</b><br/><i>Data Pipeline Lead</i><br/>Status: <b>Verified &amp; Signed</b>", table_cell_style),
            Paragraph("<b>Narmina Ibrahimova</b><br/><i>Metrics &amp; Hooks Lead</i><br/>Status: <b>Verified &amp; Signed</b>", table_cell_style),
        ],
    ]
    sig_table_1 = Table(sig_data_1, colWidths=[2.56 * inch, 2.56 * inch, 2.56 * inch])
    sig_table_1.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg_light),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    sig_data_2 = [
        [
            Paragraph("<b>Emil Ahmedli</b><br/><i>Ablation Sweeps &amp; Hardware Lead</i><br/>Status: <b>Verified &amp; Signed</b>", table_cell_style),
            Paragraph("<b>Rufet Dosteliyev</b><br/><i>Ablation Analytics, Figures &amp; Paper Lead</i><br/>Status: <b>Verified &amp; Signed</b>", table_cell_style),
        ],
    ]
    sig_table_2 = Table(sig_data_2, colWidths=[3.84 * inch, 3.84 * inch])
    sig_table_2.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg_light),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    story.append(sig_table_1)
    story.append(Spacer(1, 2))
    story.append(sig_table_2)

    doc.build(story)
    print(f"Successfully generated contribution report PDF at: {output_pdf_path}")


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "contribution_report.pdf"
    create_contribution_report(out_path)
