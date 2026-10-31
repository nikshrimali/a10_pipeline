"""Turn the run logs into pictures.

    uv run plot.py --main runs/main --accum runs/accum --ladder runs/ladder.json \
                   --gradcheck runs/gradcheck.json --lead runs/main --out plots

Every argument is optional; each one makes one figure:
    --main       trace.png      the four traces: loss, grad norm, tokens per second, MFU
    --accum      accum.png      correct against broken gradient accumulation
    --lead       lead.png       zoom on the step where grad norm moved before loss
    --ladder     ladder.png     MFU of each speed trick
    --gradcheck  gradcheck.png  numeric vs autograd gradient error against nudge size

Charts never put two different y-scales on one axis; two measures get two stacked panels that
share the x axis instead.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def read_jsonl(path):
    """Read a .jsonl file into a list of dicts."""
    return [json.loads(line) for line in open(path)]


def smooth(y, k):
    """Moving average over the previous k values, so a noisy per-step curve shows its trend."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    c = np.cumsum(np.insert(y, 0, 0.0))
    for i in range(len(y)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def plot_trace(run, out):
    """Four stacked panels sharing the step axis: loss, grad norm, tok/s, MFU.

    Thin pale lines are the raw per-step values, the solid line is a 20-step moving average.
    Held-out loss points sit on the loss panel. MFU has a dashed line at the 40% target.
    """
    recs = read_jsonl(f"{run}/metrics.jsonl")
    cfg = json.load(open(f"{run}/config.json"))
    tr = [r for r in recs if r["type"] == "train"]
    ev = [r for r in recs if r["type"] == "eval"]
    s = np.array([r["step"] for r in tr])
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    panels = [
        ("loss", "Loss: is the model learning?", "loss (nats per token)", 1.0),
        ("grad_norm", "Grad norm: trouble before the loss shows it", "pre-clip grad norm", 1.0),
        ("tok_per_s", "Tokens per second: did something quietly get slower?", "tokens / s", 1.0),
        ("mfu", "MFU: is the machine being wasted?", "MFU (%)", 100.0),
    ]
    for ax, (key, title, ylabel, scale) in zip(axes, panels):
        y = np.array([r[key] for r in tr]) * scale
        ax.plot(s, y, color=BLUE, alpha=0.25, linewidth=1)
        ax.plot(s, smooth(y, 20), color=BLUE, label="20-step average")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0].scatter([r["step"] for r in ev], [r["loss"] for r in ev], color=ORANGE, s=36, zorder=3,
                    edgecolor=SURFACE, linewidth=2, label="held-out loss")
    axes[0].legend(loc="upper right")
    axes[1].axhline(cfg["grad_clip"], color=INK2, linestyle="--", linewidth=1)
    axes[1].text(s[-1], cfg["grad_clip"], f" clip at {cfg['grad_clip']}", color=INK2, va="bottom", ha="right")
    gn = np.array([r["grad_norm"] for r in tr])
    axes[1].set_ylim(0, np.percentile(gn, 99.5) * 1.3)
    axes[3].axhline(40, color=INK2, linestyle="--", linewidth=1)
    axes[3].text(s[0], 40, " 40% target", color=INK2, va="bottom")
    if cfg.get("matmul_ceiling") and cfg.get("peak_flops"):
        c = 100 * cfg["matmul_ceiling"] / cfg["peak_flops"]
        axes[3].axhline(c, color=AQUA, linestyle=":", linewidth=1.5)
        axes[3].text(s[-1], c, f"big-matmul ceiling at run start: {c:.0f}% ", color=INK2, va="bottom", ha="right")
    axes[3].set_ylim(0, 100)
    axes[-1].set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(f"{out}/trace.png", dpi=130)
    plt.close(fig)


def plot_accum(run, out):
    """Correct vs broken accumulation: reported loss, held-out loss, and gradient error per step."""
    c = read_jsonl(f"{run}/correct.jsonl")
    b = read_jsonl(f"{run}/broken.jsonl")
    cfg = json.load(open(f"{run}/config.json"))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax = axes[0]
    ax.plot([r["step"] for r in c], smooth([r["reported_loss"] for r in c], 10), color=BLUE, label="correct: what it reports")
    ax.plot([r["step"] for r in b], smooth([r["reported_loss"] for r in b], 10), color=ORANGE, label="broken: what it reports")
    ax.plot([r["step"] for r in b], smooth([r["true_loss"] for r in b], 10), color=ORANGE, linestyle="--", linewidth=1.5,
            label="broken: its true token-weighted loss")
    ax.set_title(f"Training loss as each method reports it (micro-batch lengths {cfg['lengths']})")
    ax.set_ylabel("loss, 10-step average")
    ax.legend(loc="upper right")
    ax = axes[1]
    for rows, color, name in ((c, BLUE, "correct"), (b, ORANGE, "broken")):
        e = [r for r in rows if "eval_loss" in r]
        ax.plot([r["step"] for r in e], [r["eval_loss"] for r in e], color=color, marker="o", markersize=5, label=name)
    ax.set_title("Held-out loss at full length (256 tokens): which model is actually better?")
    ax.set_ylabel("held-out loss")
    ax.legend(loc="upper right")
    ax = axes[2]
    ax.plot([r["step"] for r in c], [r["grad_rel_error"] for r in c], color=AQUA, alpha=0.35, linewidth=1)
    ax.plot([r["step"] for r in c], smooth([r["grad_rel_error"] for r in c], 10), color=AQUA)
    ax.set_title("How wrong the broken gradient is: |g_broken - g_correct| / |g_correct|, same weights")
    ax.set_ylabel("relative error")
    ax.set_ylim(0, None)
    axes[-1].set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(f"{out}/accum.png", dpi=130)
    plt.close(fig)


def plot_lead(run, out, before=25, after=25):
    """Zoom on the clearest lead event: grad norm (top) and loss (bottom), event steps marked."""
    lead = json.load(open(f"{run}/lead.json"))
    best = lead["best"]
    if not best:
        print(f"no lead event in {run}, skipping lead.png")
        return
    tr = [r for r in read_jsonl(f"{run}/metrics.jsonl") if r["type"] == "train"]
    lo, hi = best["grad_step"] - before, best["loss_step"] + after
    tr = [r for r in tr if lo <= r["step"] <= hi]
    s = [r["step"] for r in tr]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(s, [r["grad_norm"] for r in tr], color=BLUE, marker="o", markersize=4)
    axes[0].set_title("Grad norm")
    axes[0].set_ylabel("pre-clip grad norm")
    axes[1].plot(s, [r["loss"] for r in tr], color=ORANGE, marker="o", markersize=4)
    axes[1].set_title("Loss")
    axes[1].set_ylabel("train loss")
    for ax in axes:
        ax.axvline(best["grad_step"], color=BLUE, linestyle="--", linewidth=1)
        ax.axvline(best["loss_step"], color=ORANGE, linestyle="--", linewidth=1)
    axes[0].text(best["grad_step"], axes[0].get_ylim()[1], f" grad norm moves (step {best['grad_step']})",
                 color=INK2, va="top")
    axes[1].text(best["loss_step"], axes[1].get_ylim()[1], f" loss moves (step {best['loss_step']}), "
                 f"{best['lead']} steps later", color=INK2, va="top")
    axes[-1].set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(f"{out}/lead.png", dpi=130)
    plt.close(fig)


def plot_ladder(path, out):
    """Horizontal bars, one per rung, length = MFU, labelled with tokens per second."""
    data = json.load(open(path))
    rungs = data["rungs"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = [r["rung"] for r in rungs][::-1]
    vals = [100 * r["mfu"] if "mfu" in r else 0 for r in rungs][::-1]
    bars = ax.barh(names, vals, color=BLUE, height=0.6)
    for bar, r in zip(bars, rungs[::-1]):
        label = "failed on this machine" if "error" in r else f"{100 * r['mfu']:.1f}%   {r['tok_per_s']:,.0f} tok/s"
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, label, va="center", color=INK)
    ax.axvline(40, color=INK2, linestyle="--", linewidth=1)
    ax.text(40, len(names) - 0.4, " 40% target", color=INK2)
    ax.set_xlim(0, max(vals + [40]) * 1.35)
    ax.set_xlabel("MFU (%), steps 4-10 of each rung")
    ax.set_title("The MFU ladder: each rung adds one change to the one below it")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{out}/ladder.png", dpi=130)
    plt.close(fig)


def plot_gradcheck(path, out):
    """Relative error between numeric and autograd gradient against nudge size eps, one line per dtype."""
    data = json.load(open(path))
    sw = data["sweep"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for dname, color in (("fp64", BLUE), ("fp32", ORANGE), ("bf16", AQUA)):
        ax.loglog(sw["eps"], np.maximum(sw["rel_error"][dname], 1e-12), color=color, marker="o", markersize=4, label=dname)
    ax.legend(loc="lower right")
    ax.invert_xaxis()
    ax.set_xlabel("eps, the size of the nudge (smaller to the right)")
    ax.set_ylabel("relative error, numeric vs autograd")
    ax.set_title(f"Hand-checked gradient of {sw['weight']}: there is a best nudge size per dtype")
    fig.tight_layout()
    fig.savefig(f"{out}/gradcheck.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main")
    ap.add_argument("--accum")
    ap.add_argument("--lead")
    ap.add_argument("--ladder")
    ap.add_argument("--gradcheck")
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.main:
        plot_trace(args.main, args.out)
    if args.accum:
        plot_accum(args.accum, args.out)
    if args.lead:
        plot_lead(args.lead, args.out)
    if args.ladder:
        plot_ladder(args.ladder, args.out)
    if args.gradcheck:
        plot_gradcheck(args.gradcheck, args.out)
    print(f"wrote plots to {args.out}/")


if __name__ == "__main__":
    main()
