from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "report" / "附件3_三组核心数据表.pdf"


def get_font() -> FontProperties:
    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in font_candidates:
        if path.exists():
            return FontProperties(fname=str(path))
    return FontProperties()


FONT = get_font()
plt.rcParams["axes.unicode_minus"] = False


TABLE1_COLUMNS = [
    "数据分段",
    "样本窗口区间",
    "资产数量",
    "条件因子数量",
    "总样本窗口数",
    "尾部样本数",
    "尾部样本占比",
]
TABLE1_ROWS = [
    ["训练集", "2020-12-14 至 2022-12-30", "5", "4", "498", "26", "5.22%"],
    ["验证集", "2023-01-03 至 2023-12-29", "5", "4", "242", "2", "0.83%"],
    ["测试集", "2024-01-02 至 2026-04-03", "5", "4", "544", "26", "4.78%"],
    ["合计", "2020-12-14 至 2026-04-03", "5", "4", "1284", "54", "4.21%"],
]
TABLE1_NOTE = (
    "数据采集/生成方式说明：表1依据 `configs/data.yaml`、`README.md` 以及 "
    "`notebooks/02_tail_label.ipynb` 中已运行输出整理。资产池为 5 个 ETF/指数，"
    "条件因子为 `cumret_5d`、`vol_20d`、`amount_change_5d`、`high_vol` 共 4 项。"
    "说明：平衡面板原始公共日期区间起点为 2020-11-16；由于模型采用 20 日窗口，"
    "首个可用样本窗口结束日为 2020-12-14。"
)


TABLE2_COLUMNS = [
    "统计口径",
    "样本数",
    "组合收益均值",
    "标准差",
    "VaR(5%)",
    "ES(5%)",
    "最差收益",
]
TABLE2_ROWS = [
    ["单日收益（last_day）", "544", "0.08%", "1.83%", "-2.29%", "-3.68%", "-10.83%"],
    ["20日累计收益（cum_20d）", "544", "1.75%", "8.25%", "-8.27%", "-10.78%", "-16.12%"],
]
TABLE2_NOTE = (
    "数据采集/生成方式说明：表2依据 `notebooks/03_figures.ipynb` 中对 "
    "`d_real_risk_metrics.csv` 的展示结果整理，原始统计对象为测试集真实样本。"
    "为便于中期检查材料阅读，已将 notebook 中的原始小数口径转换为百分比展示。"
)


TABLE3_COLUMNS = [
    "实验版本",
    "条件设定",
    "生成样本数",
    "生成均值",
    "生成波动率",
    "生成VaR(5%)",
    "生成ES(5%)",
    "与基线差异说明",
]
TABLE3_ROWS = [
    [
        "latest / normal_market",
        "baseline",
        "256",
        "1.143",
        "43.335",
        "-77.031",
        "-103.135",
        "作为生成基线",
    ],
    [
        "latest / normal_market",
        "cumret_5d = -1.0",
        "256",
        "0.305",
        "41.128",
        "-64.506",
        "-87.262",
        "相对基线 ES 改善 15.872",
    ],
    [
        "latest / normal_market",
        "cumret_5d = -0.5",
        "256",
        "-5.380",
        "42.027",
        "-76.351",
        "-106.124",
        "相对基线 ES 恶化 2.990",
    ],
    [
        "latest / normal_market",
        "cumret_5d = 0.0",
        "256",
        "-3.343",
        "42.389",
        "-80.184",
        "-103.473",
        "相对基线基本持平",
    ],
    [
        "latest / normal_market",
        "cumret_5d = 0.5",
        "256",
        "1.651",
        "38.500",
        "-58.368",
        "-82.713",
        "相对基线 ES 改善 20.422",
    ],
    [
        "latest / normal_market",
        "cumret_5d = 1.0",
        "256",
        "-6.471",
        "40.297",
        "-74.731",
        "-93.775",
        "相对基线 ES 改善 9.360",
    ],
]
TABLE3_NOTE = (
    "数据采集/生成方式说明：表3依据 `notebooks/03_figures.ipynb` 中对 "
    "`d_factor_sensitivity.csv` 的已运行输出整理，反映在 `normal_market` 基线下，"
    "对单一条件因子 `cumret_5d` 做 what-if 扰动后的生成样本风险指标变化。"
    "该表的数值口径保持与 D 模块导出结果一致，用于说明“条件变化会引起生成尾部风险指标变化”。"
)


def draw_cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.text(
        0.5,
        0.76,
        "附件3  三组核心数据表",
        ha="center",
        va="center",
        fontproperties=FONT,
        fontsize=24,
        weight="bold",
    )
    fig.text(
        0.5,
        0.66,
        "项目：基于条件扩散的金融尾部风险情景生成与归因中期材料",
        ha="center",
        va="center",
        fontproperties=FONT,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.54,
        "说明：本附件仅整理仓库中已运行 notebook 与配置文件可核验的数据，\n"
        "不额外补造未在现有材料中出现的实验结果。",
        ha="center",
        va="center",
        fontproperties=FONT,
        fontsize=13,
        linespacing=1.7,
    )
    fig.text(
        0.5,
        0.38,
        "表1：数据集与样本统计表\n"
        "表2：历史样本尾部风险统计表\n"
        "表3：生成样本条件实验结果对比表",
        ha="center",
        va="center",
        fontproperties=FONT,
        fontsize=15,
        linespacing=1.8,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_table_page(
    pdf: PdfPages,
    title: str,
    columns: list[str],
    rows: list[list[str]],
    note: str,
    col_widths: list[float],
    font_size: int,
    table_scale_y: float,
    note_y: float,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.22, 0.92, 0.62])
    ax.axis("off")

    fig.text(
        0.5,
        0.92,
        title,
        ha="center",
        va="center",
        fontproperties=FONT,
        fontsize=18,
        weight="bold",
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, table_scale_y)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#4a4a4a")
        cell.set_linewidth(0.8)
        cell.get_text().set_fontproperties(FONT)
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")

    fig.text(
        0.05,
        note_y,
        note,
        ha="left",
        va="top",
        fontproperties=FONT,
        fontsize=11.5,
        wrap=True,
        linespacing=1.5,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PATH) as pdf:
        draw_cover(pdf)
        draw_table_page(
            pdf=pdf,
            title="表1  数据集与样本统计表",
            columns=TABLE1_COLUMNS,
            rows=TABLE1_ROWS,
            note=TABLE1_NOTE,
            col_widths=[0.10, 0.25, 0.10, 0.13, 0.13, 0.12, 0.12],
            font_size=11.5,
            table_scale_y=2.0,
            note_y=0.15,
        )
        draw_table_page(
            pdf=pdf,
            title="表2  历史样本尾部风险统计表",
            columns=TABLE2_COLUMNS,
            rows=TABLE2_ROWS,
            note=TABLE2_NOTE,
            col_widths=[0.22, 0.10, 0.14, 0.12, 0.12, 0.12, 0.12],
            font_size=12,
            table_scale_y=2.3,
            note_y=0.18,
        )
        draw_table_page(
            pdf=pdf,
            title="表3  生成样本条件实验结果对比表",
            columns=TABLE3_COLUMNS,
            rows=TABLE3_ROWS,
            note=TABLE3_NOTE,
            col_widths=[0.15, 0.16, 0.10, 0.10, 0.10, 0.11, 0.11, 0.17],
            font_size=10,
            table_scale_y=1.8,
            note_y=0.10,
        )
    print(f"[saved] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
