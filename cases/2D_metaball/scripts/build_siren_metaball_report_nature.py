from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager


RED = "#B73545"
BLUE = "#2867A8"
GOLD = "#A47522"
INK = "#202124"
MID = "#66686B"
RULE = "#C9CACB"
PALE = "#F5F5F2"
PAPER = "#FCFCFA"

TIMES = Path(r"C:\Windows\Fonts\times.ttf")
TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
MSYH = Path(r"C:\Windows\Fonts\msyh.ttc")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch500-json", type=Path, required=True)
    p.add_argument("--plain-history", type=Path, required=True)
    p.add_argument("--siren-history", type=Path, required=True)
    p.add_argument("--canonical-history", type=Path, required=True)
    p.add_argument("--reconstruction-json", type=Path, required=True)
    p.add_argument("--reconstruction-fields", type=Path, required=True)
    p.add_argument("--selected-modes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def setup_fonts() -> None:
    for path in (TIMES, TIMES_BOLD, MSYH):
        if path.exists():
            fontManager.addfont(str(path))
    mpl.rcParams.update({
        "font.family": ["Times New Roman", "Microsoft YaHei"],
        "font.serif": ["Times New Roman"],
        "font.sans-serif": ["Microsoft YaHei"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
    })


def en(size: float, bold: bool = False) -> FontProperties:
    path = TIMES_BOLD if bold else TIMES
    if path.exists():
        return FontProperties(fname=str(path), size=size)
    return FontProperties(family="Times New Roman", size=size, weight="bold" if bold else "normal")


def mix(size: float, bold: bool = False) -> FontProperties:
    return FontProperties(
        family=["Times New Roman", "Microsoft YaHei"],
        size=size,
        weight="bold" if bold else "normal",
    )


def prop(text: str, size: float, bold: bool = False) -> FontProperties:
    return en(size, bold) if text.isascii() else mix(size, bold)


def txt(fig, x, y, text, size=8, color=INK, bold=False, ha="left", va="top", spacing=1.25):
    fig.text(x, y, text, fontproperties=prop(text, size, bold), color=color,
             ha=ha, va=va, linespacing=spacing)


def line(fig, x0, y0, x1, y1, color=RULE, lw=0.7):
    fig.lines.append(mpl.lines.Line2D([x0, x1], [y0, y1], transform=fig.transFigure,
                                     color=color, lw=lw))


def header(fig, page: int, title: str, subtitle: str):
    txt(fig, .04, .969, "BASISNET  /  2D METABALL  /  NATIVE CUDA JET3", 7.6, RED, True)
    txt(fig, .96, .969, f"{page:02d}", 9.0, RED, True, ha="right")
    txt(fig, .04, .928, title, 18.5, INK, True)
    txt(fig, .04, .884, subtitle, 8.8, MID)
    line(fig, .04, .852, .96, .852, INK, .75)


def footer(fig, text: str):
    line(fig, .04, .041, .96, .041, RULE, .55)
    txt(fig, .04, .025, text, 6.3, MID, va="center")


def minimalist_table(ax, columns, rows, widths, accent_columns=None, fontsize=7.2):
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, colWidths=widths,
                     cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, c), cell in table.get_celld().items():
        text = cell.get_text().get_text()
        cell.PAD = .028
        cell.set_facecolor(PAPER)
        cell.set_edgecolor(RULE)
        cell.set_linewidth(.48 if r else .8)
        cell.visible_edges = "B"
        cell.set_text_props(fontproperties=prop(text, fontsize if r else fontsize + .35, r == 0 or c == 0),
                            va="center", color=INK)
        if r == 0 and accent_columns and c in accent_columns:
            cell.set_text_props(fontproperties=prop(text, fontsize + .35, True),
                                color=accent_columns[c])
    return table


def page1(pdf, comparison):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    header(fig, 1, "三种 sine 网络不是同一种频谱先验",
           "普通 sin MLP、当前 spatial-SIREN 与严格 canonical SIREN 的方法和实现对照。")

    summaries = [
        (.04, RED, "SSS / ordinary sine", "原文公开实现为 PosEnc + sine MLP；\n失败的是去掉 PosEnc 的 raw 11D 消融。"),
        (.355, BLUE, "Current spatial-SIREN", "仅对 x,y 注入 omega=32；几何参数保持 raw 平滑；\n已训练至 epoch 2300，但并非 canonical SIREN。"),
        (.67, GOLD, "Strict canonical SIREN", "逐层 omega 与专用初始化配套；\n此处只做定义性对照，尚无训练结果。"),
    ]
    for x, color, title, body in summaries:
        line(fig, x, .811, x + .275, .811, color, 2.2)
        txt(fig, x, .795, title, 9.4, color, True)
        txt(fig, x, .758, body, 8.15, INK, spacing=1.36)

    columns = ["比较项", "Ordinary sin MLP / SSS family", "Current spatial-SIREN", "Strict canonical SIREN"]
    rows = [
        ["实验身份", "SSS: PE + sine hidden\nfailed arm: plain-sin + raw 11D",
         "native CUDA Jet3, FP32\nK=1024, checkpoint epoch 2300",
         "Sitzmann et al. (2020)\nnot trained in this report"],
        ["输入", "SSS: [PE_1...64(x,y), g]\nraw arm: [x_norm,y_norm,g_norm]",
         "[32*x_norm, 32*y_norm, g_norm]\nonly spatial coordinates are scaled",
         "z enters periodic layers\ngeometry should preferably be split/modulated"],
        ["隐藏层", "h_l = sin(W_l h_(l-1) + b_l)",
         "h_l = sin(W_l h_(l-1) + b_l)\nno layer-wise omega",
         "h_l = sin(omega_l * Linear(h))\ntypically omega_0 = omega_h = 30"],
        ["初始化", "SSS notebook: PyTorch Linear default\nmatched raw arm: gain/sqrt(fan-in)",
         "U(-gain/sqrt(n), +gain/sqrt(n))\ngain=2 for first/hidden layers",
         "first: U(-1/n,+1/n)\nhidden: U(-sqrt(6/n)/omega,+...)"],
        ["实验证据", "raw rank 6 -> 4; val loss 8.56e4\north approximately 1: collapsed",
         "rank 646 -> 806@500 -> 1024@2000\nval loss 2.22e4@2000",
         "method/code comparison only\nno empirical ranking claimed"],
    ]
    ax = fig.add_axes([.04, .315, .92, .355])
    minimalist_table(ax, columns, rows, [.14, .287, .287, .286], {1: RED, 2: BLUE, 3: GOLD}, 8.0)

    snippets = [
        (.04, RED, "SSS reference", "z = cat([sin/cos(2^k*x), g]), k=0...6\nh = sin(Linear(h))\nu = Linear(h)\n# explicit input frequencies"),
        (.355, BLUE, "Current implementation", "xn = 32*(2*x-1); yn = 32*(2*y-1)\nz = [xn, yn, normalize(g)]\nh = sin(Linear(h))\n# spatial bandwidth injection"),
        (.67, GOLD, "Canonical implementation", "h = sin(omega*Linear(h))\nW_first  ~ U(-1/n,+1/n)\nW_hidden ~ U(-sqrt(6/n)/omega,+...)\n# omega and initialization are coupled"),
    ]
    for x, color, title, code in snippets:
        line(fig, x, .275, x + .275, .275, color, 1.4)
        txt(fig, x, .258, title, 8.2, color, True)
        txt(fig, x, .222, code, 8.1, INK, spacing=1.36)

    footer(fig, "Sources: Shape Space Spectra official project and Colab; SIREN, Sitzmann et al., NeurIPS 2020. Strict canonical SIREN is not an empirical arm here.")
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


def page1_updated(pdf, comparison, canonical_history):
    """Compact method table with the now-trained strict canonical arm."""
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    header(fig, 1, "三种 sine 网络不是同一种频谱先验",
           "同为 raw 输入并不代表同一方法；strict canonical SIREN 现已作为独立实证臂运行。")
    latest = canonical_history[-1]
    c_epoch = int(latest["epoch"])
    c_rank = int(latest.get("validation_min_rank", latest.get("train_rank", 0)))
    c_loss = float(latest.get("validation_loss", latest["train_loss"]))
    summaries = [
        (.04, RED, "Ordinary sin MLP", "SSS 原文使用 PosEnc + sine MLP；\nplain-sin + raw 11D 是移除 PosEnc 的失败消融。"),
        (.355, BLUE, "Spatial-SIREN", "仅增强 x,y 空间带宽；几何参数保持平滑。\n该变体不是 strict canonical SIREN。"),
        (.67, GOLD, "Strict canonical SIREN", f"逐层 omega=30 + 官方初始化；全 raw 11D。\nepoch {c_epoch}: effective rank={c_rank}，发生二次坍缩。"),
    ]
    for x, color, title, body in summaries:
        line(fig, x, .811, x + .275, .811, color, 2.2)
        txt(fig, x, .795, title, 10.0, color, True)
        txt(fig, x, .758, body, 8.7, INK, spacing=1.36)

    columns = ["比较项", "Ordinary sin / raw", "Spatial-SIREN", "Strict canonical SIREN"]
    rows = [
        ["输入", "[x_norm, y_norm, g_norm]", "[32*x_norm, 32*y_norm, g_norm]", "[x_norm, y_norm, g_norm]"],
        ["周期层", "sin(Linear(h))", "sin(Linear(h))", "sin(30*Linear(h)) at every layer"],
        ["首层权重", "U(-2/sqrt(11), +...)", "same weights; spatial input x32", "U(-1/11,+1/11); official rule"],
        ["隐藏/输出权重", "gain/sqrt(fan-in)", "gain/sqrt(fan-in)", "U(-sqrt(6/n)/30,+...)"],
        ["K / precision", "1024 / all FP32", "1024 / all FP32", "1024 / all FP32"],
        ["实证结果", "rank 5 @200; collapse", "rank 725 @200; no collapse",
         f"val rank {c_rank} @ {c_epoch}; loss {c_loss:.2e}"],
    ]
    ax = fig.add_axes([.04, .335, .92, .34])
    minimalist_table(ax, columns, rows, [.14, .28, .28, .30], {1: RED, 2: BLUE, 3: GOLD}, 8.4)

    snippets = [
        (.04, RED, "Plain raw", "z = [x, y, g]\nh = sin(Linear(h))\n# insufficient spatial bandwidth"),
        (.355, BLUE, "Spatial-only frequency", "z = [32*x, 32*y, g]\nh = sin(Linear(h))\n# geometry remains smooth"),
        (.67, GOLD, "Official canonical rule", "z = [x, y, g]\nh = sin(30*Linear(h))\n# official init; not rank-preserving"),
    ]
    for x, color, title, code in snippets:
        line(fig, x, .292, x + .275, .292, color, 1.4)
        txt(fig, x, .274, title, 9.0, color, True)
        txt(fig, x, .235, code, 8.7, INK, spacing=1.4)
    txt(fig, .04, .099, "核心结论", 9.2, GOLD, True)
    txt(fig, .13, .099,
        "canonical SIREN 是优秀的单函数 INR 参数化，但其官方初始化并不保证 1024 个独立 Rayleigh branches 的 Gram 满秩。",
        9.0, INK, True)
    footer(fig, "Strict canonical result is an in-progress epoch checkpoint, not the epoch-1000 final statistic. All three arms use native CUDA Jet3 and FP32.")
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


def page_canonical_collapse(pdf, plain_history, spatial_history, canonical_history):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    latest = canonical_history[-1]
    epoch = int(latest["epoch"])
    val_rank = int(latest.get("validation_min_rank", latest.get("train_rank", 0)))
    header(fig, 3, "为什么 strict canonical SIREN 仍然 rank collapse",
           f"K=1024，但有效验证秩仅 {val_rank} @ epoch {epoch}；这是表示子空间坍缩，不是 mode 数量配置错误。")

    histories = [
        ("plain sin + raw", plain_history, RED),
        ("spatial-SIREN", spatial_history, BLUE),
        ("strict canonical", canonical_history, GOLD),
    ]
    specs = [("train_loss", "Rayleigh loss", True),
             ("train_rank", "effective raw rank", False),
             ("train_orth_error", "orthogonality error", True)]
    for i, (key, title, log) in enumerate(specs):
        ax = fig.add_axes([.055 + i*.307, .52, .275, .285])
        for label, hist, color in histories:
            rows_i = [r for r in hist if int(r["epoch"]) <= max(500, epoch)]
            ax.plot([r["epoch"] for r in rows_i], [r[key] for r in rows_i],
                    color=color, lw=1.15, label=label)
        axis_style(ax, title, log)
        if key == "train_rank":
            ax.set_ylim(-30, 1040)
        if i == 0:
            ax.legend(frameon=False, prop=en(7.8), loc="best")

    columns = ["method", "effective spatial phase coefficient", "rank @200", "orth @200"]
    rows = [
        ["plain raw", "<= 2/sqrt(11) = 0.603", "5", "1.00"],
        ["strict canonical", "<= 30/11 = 2.727", "8 train / 7 val", "1.00"],
        ["spatial-SIREN", "<= 32*2/sqrt(11) = 19.30", "725", "3.87e-8"],
    ]
    ax = fig.add_axes([.055, .347, .89, .12])
    minimalist_table(ax, columns, rows, [.20, .42, .18, .20], {0: INK}, 8.6)

    causes = [
        ("01", "11D fan-in 稀释空间权重",
         "固定几何上，9 个圆参数都是常数。它们只改变神经元相位，\n不贡献 x/y 导数；但首层仍按 fan-in=11 使用 1/11 权重界，\n因此真正决定涡量的两维空间通道被显著压窄。"),
        ("02", "canonical 带宽仍低 7.1x",
         "canonical 的最大首层空间相位系数约 2.727；\nspatial-SIREN 约 19.30，是前者 7.1 倍。高阶 Jet3 目标\n需要大量曲率方向，随机相位多样性不能替代空间频率。"),
        ("03", "共同 envelope 放大相关性",
         "1024 个分支乘以同一 C-infinity 边界包络。raw 网络带宽不足时，\n涡量主要由相同 envelope 导数项主导，Gram 谱集中到少数方向；\n官方 SIREN 初始化针对单个 INR，而非多分支正交基。"),
        ("04", "peeling 只能选择已有 span",
         "residual peeling / reorthogonalization 能从 raw span 中选方向，\n却无法生成初始化与优化从未提供的独立高频方向。于是 1024 个\n参数槽仍存在，但数值有效秩从 15 降至约 7-8。"),
    ]
    for i, (num, title, body) in enumerate(causes):
        x = .04 + (i % 2)*.47
        y = .287 - (i // 2)*.108
        color = GOLD if i in (0, 2) else BLUE
        line(fig, x, y + .016, x + .43, y + .016, color, 1.5)
        txt(fig, x, y, num, 8.4, color, True)
        txt(fig, x + .035, y, title, 8.9, INK, True)
        txt(fig, x, y - .037, body, 7.9, INK, spacing=1.32)

    txt(fig, .04, .069, "判定", 9.1, RED, True)
    txt(fig, .095, .069,
        "二次坍缩否定的是‘原样 canonical SIREN 足以保证大规模 eigenbasis 满秩’，不是 sine 激活本身。若要求 1024 个有效 modes，必须引入空间专用带宽或显式 rank-preserving 机制。",
        8.65, INK, True)
    footer(fig, "Observed ranks are numerical Gram ranks under the production threshold. Epoch-1000 training continues unchanged to preserve the strict ablation.")
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


def axis_style(ax, title, log=False):
    ax.set_title(title, fontproperties=en(10.0, True), pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.65)
    ax.grid(True, color="#D9D9D6", lw=.45, alpha=.55)
    ax.tick_params(labelsize=8.0, width=.55, length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en(8.0))
    ax.set_xlabel("epoch", fontproperties=en(8.5))
    if log:
        ax.set_yscale("log")


def page2(pdf, comparison, plain_history, siren_history):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    header(fig, 2, "plain-sin + raw 为什么在 2D metaball 中 rank 坍缩",
           "严格匹配消融：同一初始权重、同一 Jet3 后端、同一 FP32 和 K=1024；唯一变量是空间输入 omega_xy=1 或 32。")

    histories = [("plain sin + raw", plain_history, RED), ("spatial-SIREN", siren_history, BLUE)]
    plot_specs = [("train_loss", "Rayleigh loss", True),
                  ("train_rank", "effective raw rank", False),
                  ("train_orth_error", "orthogonality error", True)]
    for i, (key, title, log) in enumerate(plot_specs):
        ax = fig.add_axes([.055 + i*.307, .49, .275, .31])
        for label, hist, color in histories:
            x = np.asarray([r["epoch"] for r in hist])
            y = np.asarray([r[key] for r in hist], float)
            ax.plot(x, y, color=color, lw=1.05, label=label)
        axis_style(ax, title, log)
        if key == "train_rank":
            ax.set_ylim(-30, 1040)
        if i == 0:
            ax.legend(frameon=False, prop=en(7.7), loc="best")

    plain = next(r for r in comparison["results"] if r["method"] == "plain sin")
    siren = next(r for r in comparison["results"] if r["method"] == "spatial-SIREN")
    line(fig, .04, .444, .96, .444, RULE, .65)
    txt(fig, .04, .422, "Matched evidence", 8.7, MID, True)
    txt(fig, .165, .422,
        f"plain raw: rank {plain['initial_rank']} -> {plain['epoch500_validation_rank']}, "
        f"val loss {plain['epoch500_validation_loss']:.2e}, orth {plain['epoch500_validation_orth_error']:.2e}",
        8.65, RED, True)
    txt(fig, .585, .422,
        f"spatial-SIREN: rank {siren['initial_rank']} -> {siren['epoch500_validation_rank']}, "
        f"val loss {siren['epoch500_validation_loss']:.2e}, orth {siren['epoch500_validation_orth_error']:.2e}",
        8.65, BLUE, True)

    analyses = [
        ("01", "带宽缺口", "固定几何时，9 个圆参数均为常数；\n真正变化的只有 x,y。raw 坐标配 gain\n初始化后，多数 sine 相位位于近线性区，\n空间频率与曲率不足。"),
        ("02", "Jet3 放大缺口", "纯涡量 Rayleigh 目标依赖空间二、三阶\n导数。近线性 sine 的高阶导数弱且相关；\nJet3 所见的导数子空间先于前向值域\n发生贫化。"),
        ("03", "共同 envelope 诱发共线", "所有 mode 乘同一 C-infinity 边界包络。\n内部振荡不足时，涡量由共同 envelope\n导数主导，Gram 空间迅速只剩少数\n共享方向。"),
        ("04", "peeling 不能创造秩", "residual peeling 只能从已有 raw span\n选取并正交化方向；后续 mode 接到近零\n或共线残差，因此 rank 6 -> 4，\north 约为 1。"),
    ]
    for i, (num, title, body) in enumerate(analyses):
        x = .04 + i*.23
        color = RED if i in (0, 2) else BLUE
        line(fig, x, .356, x + .205, .356, color, 1.5)
        txt(fig, x, .339, num, 8.0, color, True)
        txt(fig, x + .035, .339, title, 9.0, INK, True)
        txt(fig, x, .295, body, 8.15, INK, spacing=1.38)

    txt(fig, .04, .104, "结论", 8.2, RED, True)
    txt(fig, .095, .104,
        "崩溃证明的是 raw-input 频率机制缺失，而不是 sine 激活本身不适合。移除 SSS PosEnc 后，必须补上空间带宽机制；当前 omega_xy=32 正是这一补偿。",
        8.7, INK, True)
    footer(fig, "Interpretation controls backend, precision, initialization checkpoint, geometry distribution and K. The failure is attributed to spatial bandwidth, not CUDA Jet3 or FP32.")
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


def gridify(valid, values):
    shape = valid.shape + values.shape[1:]
    out = np.full(shape, np.nan, dtype=float)
    out[valid] = values
    return out


def field_ax(fig, rect, data, valid, title, cmap, vmin, vmax):
    ax = fig.add_axes(rect)
    cm = mpl.colormaps[cmap].copy()
    cm.set_bad(PAPER)
    ax.imshow(data, origin="lower", cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.contour(valid.astype(float), levels=[.5], colors="#505050", linewidths=.45, origin="lower")
    ax.set_title(title, fontproperties=en(8.5, True), pad=2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def page3(pdf, recon, fields, page_number=3):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    header(fig, page_number, "epoch 2300：全 K=1024 随机未见流场重构",
           "同一预注册随机几何和流场、同一 70/30 节点划分；spatial-SIREN 与逐几何 Laplacian 公平拟合。")
    valid = fields["valid"]
    gt_o = gridify(valid, fields["gt_omega"])
    si_o = gridify(valid, fields["siren_omega"])
    la_o = gridify(valid, fields["laplacian_omega"])
    gt_v = gridify(valid, fields["gt_velocity"])
    si_v = gridify(valid, fields["siren_velocity"])
    la_v = gridify(valid, fields["laplacian_velocity"])
    gt_s, si_s, la_s = (np.linalg.norm(x, axis=-1) for x in (gt_v, si_v, la_v))
    si_oe, la_oe = np.abs(si_o - gt_o), np.abs(la_o - gt_o)
    si_ve, la_ve = np.linalg.norm(si_v - gt_v, axis=-1), np.linalg.norm(la_v - gt_v, axis=-1)
    om = np.nanpercentile(np.abs(gt_o), 99)
    sp = np.nanpercentile(np.concatenate([gt_s[valid], si_s[valid], la_s[valid]]), 99)
    oe = np.nanpercentile(np.concatenate([si_oe[valid], la_oe[valid]]), 99)
    ve = np.nanpercentile(np.concatenate([si_ve[valid], la_ve[valid]]), 99)
    neural, lap = recon["methods"]
    titles1 = ["Ground truth", f"spatial-SIREN  e_omega={neural['test_vorticity_relative_l2']:.3f}",
               f"Laplacian  e_omega={lap['test_vorticity_relative_l2']:.3f}", "SIREN |error|", "Laplacian |error|"]
    titles2 = ["Ground truth", f"spatial-SIREN  e_u={neural['test_velocity_relative_l2']:.3f}",
               f"Laplacian  e_u={lap['test_velocity_relative_l2']:.3f}", "SIREN vector |error|", "Laplacian vector |error|"]
    data1 = [(gt_o, "RdBu_r", -om, om), (si_o, "RdBu_r", -om, om), (la_o, "RdBu_r", -om, om),
             (si_oe, "magma", 0, oe), (la_oe, "magma", 0, oe)]
    data2 = [(gt_s, "Blues", 0, sp), (si_s, "Blues", 0, sp), (la_s, "Blues", 0, sp),
             (si_ve, "magma", 0, ve), (la_ve, "magma", 0, ve)]
    for row, (items, titles) in enumerate(((data1, titles1), (data2, titles2))):
        y = .615 if row == 0 else .385
        for i, ((data, cmap, vmin, vmax), title) in enumerate(zip(items, titles)):
            field_ax(fig, [.055 + i*.184, y, .155, .19], data, valid, title, cmap, vmin, vmax)
    txt(fig, .027, .711, "Vorticity", 8.0, MID, True, ha="center", va="center")
    txt(fig, .027, .478, "Speed", 8.0, MID, True, ha="center", va="center")
    txt(fig, .04, .352, "Shared physical-field scales within each row; shared error scales within each error pair. Full K=1024 modes are used.", 7.25, MID)

    columns = ["method", "train e_omega", "test e_omega", "train e_u", "test e_u", "rank"]
    rows = [["spatial-SIREN @2300", f"{neural['train_vorticity_relative_l2']:.3f}", f"{neural['test_vorticity_relative_l2']:.3f}",
             f"{neural['train_velocity_relative_l2']:.3f}", f"{neural['test_velocity_relative_l2']:.3f}", str(neural['fit_gram_effective_rank'])],
            ["geometry-aware Laplacian", f"{lap['train_vorticity_relative_l2']:.3f}", f"{lap['test_vorticity_relative_l2']:.3f}",
             f"{lap['train_velocity_relative_l2']:.3f}", f"{lap['test_velocity_relative_l2']:.3f}", str(lap['fit_gram_effective_rank'])]]
    ax = fig.add_axes([.04, .105, .54, .19])
    minimalist_table(ax, columns, rows, [.30, .14, .14, .14, .14, .10], {0: INK}, 8.0)
    txt(fig, .04, .087, "Coefficients fitted from vorticity only; matched relative ridge 1e-6.", 7.0, MID)

    vort_pct = 100*(neural['test_vorticity_relative_l2']/lap['test_vorticity_relative_l2'] - 1)
    vel_pct = 100*(neural['test_velocity_relative_l2']/lap['test_velocity_relative_l2'] - 1)
    line(fig, .625, .295, .96, .295, BLUE, 1.8)
    txt(fig, .625, .277, "Result", 9.2, BLUE, True)
    result = (f"Vorticity: {neural['test_vorticity_relative_l2']:.3f} vs {lap['test_vorticity_relative_l2']:.3f} "
              f"(SIREN {vort_pct:+.1f}%).\n"
              f"Velocity: {neural['test_velocity_relative_l2']:.3f} vs {lap['test_velocity_relative_l2']:.3f} "
              f"(SIREN {vel_pct:+.1f}%).\n\n"
              "两者在该随机场上近似持平。\n"
              "Laplacian 对每个新几何重新组装矩阵，\n"
              "并求 1024 个特征向量；神经基一次训练后\n"
              "可跨几何族查询。strict canonical arm 尚未训练。")
    txt(fig, .625, .242, result, 8.45, INK, spacing=1.31)
    footer(fig, f"Reproducibility: epoch={recon['checkpoint_epoch']}; grid={recon['grid_resolution']}^2; fluid nodes={recon['fluid_nodes']}; fit/test={recon['fit_nodes']}/{recon['held_out_test_nodes']}; all neural compute FP32.")
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


def page_modes(pdf, page_number, title, subtitle, valid, selected, omega, velocity, accent):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    header(fig, page_number, title, subtitle)
    txt(fig, .04, .821, "FIRST 16 MODES", 9.2, accent, True)
    line(fig, .155, .813, .96, .813, accent, 1.25)
    txt(fig, .04, .415, "LAST 16 MODES", 9.2, accent, True)
    line(fig, .155, .407, .96, .407, accent, 1.25)

    n = valid.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    row_y = [.625, .455, .219, .049]
    for slot, mode_index in enumerate(selected):
        row, col = divmod(slot, 8)
        ax = fig.add_axes([.055 + col*.114, row_y[row], .101, .145])
        w = np.full(valid.shape, np.nan, dtype=float)
        u = np.full(valid.shape, np.nan, dtype=float)
        v = np.full(valid.shape, np.nan, dtype=float)
        w[valid] = omega[:, slot]
        u[valid] = velocity[:, 0, slot]
        v[valid] = velocity[:, 1, slot]
        vmax = max(float(np.nanpercentile(np.abs(w), 99)), 1.0e-12)
        cmap = mpl.colormaps["RdBu_r"].copy()
        cmap.set_bad(PAPER)
        ax.imshow(w, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                  interpolation="nearest")
        ax.contour(valid.astype(float), levels=[.5], colors="#4B4B4B",
                   linewidths=.38, origin="lower")
        speed = np.sqrt(u*u + v*v)
        qscale = max(float(np.nanpercentile(speed, 95)), 1.0e-12)
        step = 14
        qx, qy = xx[::step, ::step], yy[::step, ::step]
        qu, qv = u[::step, ::step]/qscale, v[::step, ::step]/qscale
        ax.quiver(qx, qy, qu, qv, color="#232323", alpha=.68, scale=10.5,
                  width=.007, headwidth=3.0, headlength=3.8, pivot="mid")
        ax.set_title(f"mode {int(mode_index)+1}", fontproperties=en(8.5, True),
                     color=INK, pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    footer(fig, "Each tile uses an independent symmetric vorticity scale; arrows show the corresponding divergence-free velocity direction after per-mode normalization.")
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


def main():
    a = args()
    setup_fonts()
    comparison = json.loads(a.epoch500_json.read_text(encoding="utf-8"))
    plain = [r for r in json.loads(a.plain_history.read_text(encoding="utf-8")) if int(r["epoch"]) <= 500]
    siren = [r for r in json.loads(a.siren_history.read_text(encoding="utf-8")) if int(r["epoch"]) <= 500]
    canonical = json.loads(a.canonical_history.read_text(encoding="utf-8"))
    recon = json.loads(a.reconstruction_json.read_text(encoding="utf-8"))
    with np.load(a.reconstruction_fields, allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    with np.load(a.selected_modes, allow_pickle=False) as z:
        modes = {k: z[k] for k in z.files}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(a.output, metadata={"Title": "2D Metaball sine-network ablation", "Author": "BasisNet"}) as pdf:
        page1_updated(pdf, comparison, canonical)
        page2(pdf, comparison, plain, siren)
        page_canonical_collapse(pdf, plain, siren, canonical)
        page3(pdf, recon, fields, 4)
        page_modes(pdf, 5, "spatial-SIREN：前 16 与最后 16 个 vector modes",
                   "epoch 2300，未见几何 token 3,000,001；涡量底图叠加速度方向。",
                   modes["valid"], modes["selected_indices"], modes["neural_omega"],
                   modes["neural_velocity"], BLUE)
        page_modes(pdf, 6, "Laplacian：前 16 与最后 16 个 vector modes",
                   "同一未见几何上的 geometry-aware Dirichlet streamfunction Laplacian；K=1024。",
                   modes["valid"], modes["selected_indices"], modes["laplacian_omega"],
                   modes["laplacian_velocity"], RED)
    print(a.output)


if __name__ == "__main__":
    main()
