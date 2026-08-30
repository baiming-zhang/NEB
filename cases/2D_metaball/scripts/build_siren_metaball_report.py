from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from PIL import Image


RED = "#C43C4E"
BLUE = "#2C6DB2"
NAVY = "#17365D"
INK = "#171717"
MID = "#626262"
LIGHT = "#F4F5F7"
GRID = "#D9DDE3"
GOLD = "#B07D2C"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch500-json", type=Path, required=True)
    p.add_argument("--epoch500-figure", type=Path, required=True)
    p.add_argument("--reconstruction-json", type=Path, required=True)
    p.add_argument("--reconstruction-figure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def configure_fonts() -> FontProperties:
    chinese_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if chinese_path.exists():
        prop = FontProperties(fname=str(chinese_path))
        mpl.rcParams["font.sans-serif"] = [prop.get_name(), "Arial", "DejaVu Sans"]
    else:
        prop = FontProperties(family="DejaVu Sans")
        mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.edgecolor": INK,
        }
    )
    return prop


def page_header(fig, number: int, title: str, subtitle: str | None = None) -> None:
    fig.text(0.035, 0.958, f"{number:02d}", fontsize=10, weight="bold", color=RED, va="top")
    fig.text(0.072, 0.958, title, fontsize=18, weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.073, 0.921, subtitle, fontsize=8.5, color=MID, va="top")
    fig.lines.append(
        mpl.lines.Line2D([0.035, 0.965], [0.902, 0.902], transform=fig.transFigure, color=INK, lw=0.8)
    )


def footer(fig, text: str) -> None:
    fig.lines.append(
        mpl.lines.Line2D([0.035, 0.965], [0.043, 0.043], transform=fig.transFigure, color=GRID, lw=0.65)
    )
    fig.text(0.035, 0.017, text, fontsize=6.3, color=MID, va="bottom")


def rounded_box(fig, xywh, facecolor=LIGHT, edgecolor="none", radius=0.012, lw=0.8):
    x, y, w, h = xywh
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    fig.patches.append(box)
    return box


def fig_text(fig, x, y, text, *, size=8, color=INK, weight="normal", va="top", ha="left", linespacing=1.3):
    fig.text(
        x, y, text, fontsize=size, color=color, weight=weight,
        va=va, ha=ha, linespacing=linespacing,
    )


def page_one(pdf: PdfPages, epoch500: dict, font_prop: FontProperties) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    page_header(
        fig, 1,
        "普通 sin MLP、当前 spatial-SIREN 与严格 canonical SIREN",
        "同一个 sin(·) 符号并不代表同一个频谱先验：输入频率、逐层尺度与初始化共同决定 Jet3 导数空间。",
    )

    rounded_box(fig, (0.035, 0.825, 0.93, 0.058), facecolor="#EDF3FA", edgecolor="#BCD0E7")
    fig_text(fig, 0.052, 0.866, "核心判断", size=8.5, color=BLUE, weight="bold")
    fig_text(
        fig, 0.135, 0.866,
        "SSS 原始公开实现是 PosEnc + 普通 sine MLP；本次崩溃的是去掉 PosEnc 的 raw 11D 消融。"
        "当前方法只对空间坐标注入 ω=32，已解决 rank 坍缩，但它不是严格 canonical SIREN。",
        size=8.4,
    )

    ax = fig.add_axes([0.035, 0.315, 0.93, 0.49])
    ax.axis("off")
    columns = [
        "比较项",
        "普通 sin MLP\n（SSS family）",
        "当前 spatial-SIREN\n（已训练）",
        "严格 canonical SIREN\n（定义性对照）",
    ]
    rows = [
        [
            "公开/实验身份",
            "SSS 公开代码：PE + sine hidden\n失败臂：plain-sin + raw 11D",
            "本次 metaball 主训练臂\nCUDA Jet3，epoch 2300",
            "Sitzmann et al. 2020\n本项目尚未训练此臂",
        ],
        [
            "网络输入 z",
            "SSS: [PE_1...64(x,y), g]\n失败臂: [x_norm, y_norm, g_norm]",
            "[32*x_norm, 32*y_norm, g_norm]\n只放大空间，几何保持平滑",
            "通常对全部输入使用 z\n几何条件宜拆分/调制",
        ],
        [
            "隐藏层",
            "h_l = sin(W_l h_(l-1) + b_l)",
            "h_l = sin(W_l h_(l-1) + b_l)\n无逐层 omega_l",
            "h_l = sin(omega_l * Linear(h))\n常用 omega_0=omega_h=30",
        ],
        [
            "初始化",
            "SSS notebook: PyTorch Linear 默认\n匹配失败臂: gain/√fan-in",
            "U(-gain/sqrt(n), +gain/sqrt(n))\ngain=2（首/隐层）",
            "首层 U(-1/n,+1/n)\n隐层 U(-sqrt(6/n)/omega,+...)",
        ],
        [
            "频率从哪里来",
            "原 SSS：显式 PosEnc\nraw 消融：没有补偿 → 饥饿",
            "输入端空间尺度 omega_xy=32\n工程化 bandwidth injection",
            "逐层 omega + 方差保持初始化\n严格的周期激活分布控制",
        ],
        [
            "metaball 证据",
            "raw: rank 6→4；val loss 8.56×10⁴\north≈1（坍缩）",
            "rank 646→806@500→1024@2000\nval loss 2.22×10⁴@2000",
            "未运行：不能宣称优劣\n仅列方法与代码差异",
        ],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0, 0, 1, 1],
        colWidths=[0.145, 0.285, 0.285, 0.285],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.3)
    for (r, c), cell in table.get_celld().items():
        cell.PAD = 0.025
        cell.set_text_props(fontproperties=font_prop, va="center")
        if r == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", weight="bold", fontproperties=font_prop, fontsize=7.5)
            cell.set_edgecolor("white")
            cell.set_linewidth(0.0)
        else:
            cell.set_facecolor("white" if r % 2 else "#F7F7F7")
            cell.set_edgecolor("#D7D7D7")
            cell.set_linewidth(0.45)
            if c == 0:
                cell.set_text_props(weight="bold", fontproperties=font_prop, fontsize=7.2)
        if r == 6 and c == 1:
            cell.set_text_props(color=RED, weight="bold", fontproperties=font_prop, fontsize=7.2)
        if r == 6 and c == 2:
            cell.set_text_props(color=BLUE, weight="bold", fontproperties=font_prop, fontsize=7.2)

    box_y, box_h, gap = 0.085, 0.205, 0.014
    widths = [0.304, 0.304, 0.304]
    xs = [0.035, 0.035 + widths[0] + gap, 0.035 + 2 * (widths[0] + gap)]
    titles = ["SSS 官方示例（概念化）", "本次 spatial-SIREN（实际实现）", "canonical SIREN（官方定义）"]
    colors = [MID, BLUE, GOLD]
    snippets = [
        "z = cat([sin/cos(2^k*x), g])   k=0...6\n"
        "h1 = sin(Linear(z))\n"
        "…  h4 = sin(Linear(h3))\n"
        "u  = Linear(h4)\n"
        "# 原方法不是 raw-input arm",
        "xn = 32*(2*x-1); yn = 32*(2*y-1)\n"
        "gn = normalize(circle_parameters)\n"
        "z = [xn, yn, gn]               # 11D\n"
        "h = sin(Linear(h))             # layer omega=1\n"
        "W ~ U(-gain/sqrt(fan_in), +...)",
        "h = sin(omega*Linear(h))       # omega~30\n"
        "W_first  ~ U(-1/n,+1/n)\n"
        "W_hidden ~ U(-sqrt(6/n)/omega,+...)\n"
        "# 逐层尺度与初始化必须配套\n"
        "# 参数条件最好与空间支路分离",
    ]
    for x, w, title, color, snippet in zip(xs, widths, titles, colors, snippets):
        rounded_box(fig, (x, box_y, w, box_h), facecolor="#FAFAFA", edgecolor="#DDDDDD", radius=0.008)
        fig_text(fig, x + 0.012, box_y + box_h - 0.018, title, size=8.0, color=color, weight="bold")
        fig_text(fig, x + 0.012, box_y + box_h - 0.052, snippet, size=7.15, color=INK, linespacing=1.48)

    footer(
        fig,
        "Sources: Shape Space Spectra official project + Colab (Fourier PE 1–64, sine hidden layers); "
        "SIREN paper arXiv:2006.09661 + official modules.py. Strict canonical arm is not an empirical result here.",
    )
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def page_two(pdf: PdfPages, epoch500: dict, evidence_path: Path) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    page_header(
        fig, 2,
        "为什么 plain-sin + raw 在 2D metaball 中发生 rank 坍缩",
        "严格匹配消融：同一初始 checkpoint hash、同一 native CUDA Jet3、同一 K=1024、同一 FP32；唯一变量是空间输入 ωxy=1 vs 32。",
    )

    with Image.open(evidence_path) as im:
        w, h = im.size
        crop = im.crop((0, 80, w, int(h * 0.83)))
        crop_array = mpl.image.pil_to_array(crop)
    ax_img = fig.add_axes([0.035, 0.345, 0.93, 0.54])
    ax_img.imshow(crop_array)
    ax_img.axis("off")

    plain = next(x for x in epoch500["results"] if x["method"] == "plain sin")
    siren = next(x for x in epoch500["results"] if x["method"] == "spatial-SIREN")
    fig_text(
        fig, 0.04, 0.335,
        f"实测：plain-sin raw rank {plain['initial_rank']}→{plain['epoch500_validation_rank']}，"
        f"val loss {plain['epoch500_validation_loss']:.2e}，orth {plain['epoch500_validation_orth_error']:.2e}；"
        f"spatial-SIREN rank {siren['initial_rank']}→{siren['epoch500_validation_rank']}，"
        f"val loss {siren['epoch500_validation_loss']:.2e}，orth {siren['epoch500_validation_orth_error']:.2e}。",
        size=7.6, color=MID,
    )

    y, h, gap = 0.076, 0.225, 0.013
    col_w = (0.93 - 3 * gap) / 4
    titles = ["1 · 带宽缺口", "2 · Jet3 放大缺口", "3 · envelope 诱发共线", "4 · peeling 无法造秩"]
    bodies = [
        "在固定几何上，9 个圆参数均为常数。\n"
        "真正变化的只有 x,y。raw 坐标配 gain\n"
        "初始化，使多数 sine 相位停在近线性区；\n"
        "空间频率与曲率不足。",
        "纯涡量 Rayleigh 目标依赖空间二、三阶\n"
        "导数。近线性 sine 的高阶导数既弱又\n"
        "高度相关：问题不在前向值域，而在\n"
        "Jet3 所见的导数子空间已贫化。",
        "所有 mode 还乘同一个 C-infinity 边界/\n"
        "障碍 envelope。缺少内部振荡时，涡量\n"
        "主要由共同 envelope 的导数主导，Gram\n"
        "谱迅速变成少数共享方向。",
        "residual peeling 只能在已有 raw span 中\n"
        "正交化；后续 mode 接到近零/共线残差。\n"
        "正交化可选方向、不能创造方向，因而\n"
        "rank 6->4、orth≈1。",
    ]
    for i, (title, body) in enumerate(zip(titles, bodies)):
        x = 0.035 + i * (col_w + gap)
        rounded_box(fig, (x, y, col_w, h), facecolor="#FAFAFA", edgecolor="#DEDEDE", radius=0.007)
        fig_text(fig, x + 0.011, y + h - 0.017, title, size=8.0, color=RED if i in (0, 2) else BLUE, weight="bold")
        fig_text(fig, x + 0.011, y + h - 0.055, body, size=7.15, color=INK, linespacing=1.43)

    footer(
        fig,
        "Interpretation: collapse is not evidence that sine activations are intrinsically unsuitable. It is evidence that removing SSS positional encoding requires a replacement spatial-frequency mechanism. Backend/precision are controlled here.",
    )
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def page_three(pdf: PdfPages, recon: dict, reconstruction_path: Path) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    page_header(
        fig, 3,
        "epoch 2300：全 K=1024 随机未见流场重构 vs geometry-aware Laplacian",
        "随机种子 20260819、未见几何 token 3,000,001；24 项随机流函数 GT；同一 70% 拟合 / 30% 留出节点；不是挑选有利样本。",
    )

    image = mpimg.imread(reconstruction_path)
    ax_img = fig.add_axes([0.035, 0.345, 0.93, 0.545])
    ax_img.imshow(image)
    ax_img.axis("off")

    methods = recon["methods"]
    neural = methods[0]
    lap = methods[1]
    ax_table = fig.add_axes([0.035, 0.105, 0.53, 0.205])
    ax_table.axis("off")
    columns = ["method", "train eω", "test eω", "train eu", "test eu", "rank"]
    rows = [
        [
            "spatial-SIREN @2300",
            f"{neural['train_vorticity_relative_l2']:.3f}",
            f"{neural['test_vorticity_relative_l2']:.3f}",
            f"{neural['train_velocity_relative_l2']:.3f}",
            f"{neural['test_velocity_relative_l2']:.3f}",
            str(neural["fit_gram_effective_rank"]),
        ],
        [
            "geometry-aware Laplacian",
            f"{lap['train_vorticity_relative_l2']:.3f}",
            f"{lap['test_vorticity_relative_l2']:.3f}",
            f"{lap['train_velocity_relative_l2']:.3f}",
            f"{lap['test_velocity_relative_l2']:.3f}",
            str(lap["fit_gram_effective_rank"]),
        ],
    ]
    table = ax_table.table(
        cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
        loc="upper left", bbox=[0, 0.18, 1, 0.82],
        colWidths=[0.29, 0.14, 0.14, 0.14, 0.14, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (r, c), cell in table.get_celld().items():
        cell.PAD = 0.02
        cell.set_linewidth(0)
        if r == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", weight="bold", fontsize=6.8)
        else:
            cell.set_facecolor("white")
            cell.visible_edges = "B"
            cell.set_edgecolor("#9B9B9B")
            cell.set_linewidth(0.55)
            if c == 0:
                cell.set_text_props(ha="left", weight="bold", color=BLUE if r == 1 else INK)
    fig_text(
        fig, 0.04, 0.123,
        "eω / eu = held-in or held-out relative L2 error. Coefficients are fitted from vorticity only with matched relative ridge 1e-6.",
        size=6.6, color=MID,
    )

    rounded_box(fig, (0.59, 0.105, 0.375, 0.205), facecolor="#F7F8FA", edgecolor="#D8DCE2", radius=0.008)
    vort_delta = 100 * (neural["test_vorticity_relative_l2"] / lap["test_vorticity_relative_l2"] - 1)
    vel_delta = 100 * (neural["test_velocity_relative_l2"] / lap["test_velocity_relative_l2"] - 1)
    fig_text(fig, 0.605, 0.289, "结论与边界", size=8.4, color=NAVY, weight="bold")
    conclusion = (
        f"• 留出涡量：SIREN {neural['test_vorticity_relative_l2']:.3f} vs Laplacian {lap['test_vorticity_relative_l2']:.3f}\n"
        f"  （SIREN 高 {vort_delta:.1f}%）。\n"
        f"• 留出速度：SIREN {neural['test_velocity_relative_l2']:.3f} vs Laplacian {lap['test_velocity_relative_l2']:.3f}\n"
        f"  （SIREN 低 {abs(vel_delta):.1f}%）；两者在该随机场上近似持平。\n"
        "• Laplacian 对每个新几何重组矩阵并求 1024 个特征向量；\n"
        "  当前网络一次训练，可跨三圆几何族查询。\n"
        "• 这是最新冻结 checkpoint 的单个预注册随机场；strict\n"
        "  canonical arm 尚未训练，也不是 5000 epoch 最终统计。"
    )
    fig_text(fig, 0.605, 0.259, conclusion, size=7.15, color=INK, linespacing=1.35)

    footer(
        fig,
        f"Reproducibility: checkpoint epoch={recon['checkpoint_epoch']}; grid={recon['grid_resolution']}²; fluid nodes={recon['fluid_nodes']}; "
        f"fit/test={recon['fit_nodes']}/{recon['held_out_test_nodes']}; all neural compute FP32. Laplacian: 5-point FD, ψ=0 on outer/obstacle boundaries, per-geometry eigsh.",
    )
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    font_prop = configure_fonts()
    epoch500 = json.loads(args.epoch500_json.read_text(encoding="utf-8"))
    recon = json.loads(args.reconstruction_json.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output, metadata={
        "Title": "2D Metaball: sine MLP, spatial-SIREN, canonical SIREN and Laplacian",
        "Author": "BasisNet ablation report",
        "Subject": "Native CUDA Jet3, FP32, K=1024",
    }) as pdf:
        page_one(pdf, epoch500, font_prop)
        page_two(pdf, epoch500, args.epoch500_figure)
        page_three(pdf, recon, args.reconstruction_figure)
    print(args.output)


if __name__ == "__main__":
    main()
