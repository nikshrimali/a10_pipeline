"""Check backward() by hand: nudge one weight, watch the loss move, compare with the autograd gradient.

    uv run gradcheck.py --data data/fwe

For a weight w, the gradient dL/dw says how fast the loss changes when w changes. We can
measure that without calculus:

    numeric gradient = (L(w + eps) - L(w - eps)) / (2 * eps)        "central difference"

If autograd is right, the two numbers agree. This script checks five single weights from
different parts of the model, first in fp64 (where they should agree to 8 or more digits), then
in fp32 and bf16, where rounding eats the agreement. A final sweep over eps shows why there is a
best eps: too big and the formula's approximation error dominates, too small and the loss
difference drowns in rounding noise.

Runs on the CPU (MPS has no fp64). Writes runs/gradcheck.json for the guide and plot.py.
"""

import argparse
import copy
import json
import math
import os

import numpy as np
import torch

from model import GPT, GPTConfig, token_loss
from train import get_batch, load_tokens

DTYPES = {"fp64": torch.float64, "fp32": torch.float32, "bf16": torch.bfloat16}
# The eps a beginner would reach for in each dtype. fp64's comes from the sweep's flat bottom.
# fp32's 1e-3 looks reasonable and fails, which is the lesson; the sweep shows its real best.
DEFAULT_EPS = {"fp64": 1e-5, "fp32": 1e-3, "bf16": 1e-1}


def pick_weights(model, x):
    """Choose five single weights to check, one from each kind of layer.

    Args:
        model: the GPT.
        x: (B, T) the input batch. The token embedding row we check belongs to a token that is
            in this batch, otherwise its input-side gradient would be zero.

    Returns:
        A list of (parameter name, index tuple, plain description).
    """
    tid = int(x[0, 0])
    return [
        ("tok_emb.weight", (tid, 5), f"embedding of token {tid}, channel 5 (also used by the output head)"),
        ("pos_emb.weight", (3, 7), "position 3's vector, channel 7"),
        ("blocks.0.attn.qkv.weight", (10, 20), "a query weight in block 0's attention"),
        ("blocks.3.mlp.fc.weight", (100, 50), "a weight in block 3's MLP"),
        ("blocks.5.ln2.weight", (17,), "a LayerNorm gain in block 5"),
    ]


def loss_at(model, x, y):
    """Mean next-token loss of `model` on (x, y), as a Python float, without building a graph."""
    with torch.no_grad():
        return token_loss(model(x), y).item()


def check(model, x, y, picks, eps):
    """Compare numeric and autograd gradients for each picked weight.

    Args:
        model: a GPT already in the dtype to test.
        x, y: one batch, on the CPU.
        picks: output of pick_weights().
        eps: the nudge size.

    Returns:
        A list of dicts: name, description, autograd and numeric gradients, relative error,
        digits of agreement, and the nudge the dtype could actually store.
    """
    params = dict(model.named_parameters())
    model.zero_grad()
    token_loss(model(x), y).backward()
    rows = []
    for name, idx, desc in picks:
        p = params[name]
        auto = p.grad[idx].item()
        with torch.no_grad():
            w = p[idx].clone()
            p[idx] = w + eps
            w_plus = p[idx].item()
            l_plus = loss_at(model, x, y)
            p[idx] = w - eps
            w_minus = p[idx].item()
            l_minus = loss_at(model, x, y)
            p[idx] = w
        numeric = (l_plus - l_minus) / (2 * eps)
        rel = abs(numeric - auto) / max(abs(numeric), abs(auto), 1e-30)
        # The smallest loss change this dtype can resolve near the current loss, against the
        # change we expect from the nudge. When expected < resolvable, the numeric gradient is noise.
        resolvable = torch.finfo(p.dtype).eps * l_plus
        rows.append({"expected_change": 2 * eps * abs(auto), "resolvable_change": resolvable, "name": name, "index": list(idx), "desc": desc, "weight": w.item(), "autograd": auto,
                     "numeric": numeric, "rel_error": rel, "digits": -math.log10(max(rel, 1e-17)),
                     "eps": eps, "stored_nudge": (w_plus - w_minus) / 2, "loss_plus": l_plus, "loss_minus": l_minus})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--out", default="runs/gradcheck.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    meta = json.load(open(f"{args.data}/meta.json"))
    base = GPT(GPTConfig(vocab=meta["vocab_size"], context=256, attn="manual"))
    x, y = get_batch(load_tokens(args.data, "train"), range(args.batch), args.seq_len, torch.device("cpu"))
    picks = pick_weights(base, x)
    print(f"batch: {args.batch} sequences x {args.seq_len} tokens, model in eval-free train mode (no dropout exists)\n")

    results = {}
    for dname, dtype in DTYPES.items():
        model = copy.deepcopy(base).to(dtype)
        eps = DEFAULT_EPS[dname]
        rows = check(model, x, y, picks, eps)
        results[dname] = rows
        print(f"== {dname}, eps = {eps:g} ==")
        print(f"{'weight':<34} {'autograd':>14} {'numeric':>14} {'rel error':>10} {'digits':>6}")
        for r in rows:
            label = f"{r['name']}{tuple(r['index'])}"
            print(f"{label:<34} {r['autograd']:>14.9f} {r['numeric']:>14.9f} {r['rel_error']:>10.2e} {r['digits']:>6.1f}")
        print(f"  loss {rows[0]['loss_plus']:.4f}: {dname} can resolve loss changes of about {rows[0]['resolvable_change']:.1e}; "
              f"the nudges should move it by {min(r['expected_change'] for r in rows):.1e} to {max(r['expected_change'] for r in rows):.1e}")
        if dname == "bf16":
            print("bf16 stores the weight with only 8 significant bits, so the nudge it actually applied was:")
            for r in rows:
                print(f"  {r['name']}{tuple(r['index'])}: asked for {r['eps']:g}, stored {r['stored_nudge']:.6g}")
        print()

    # Sweep eps for one weight to show the U-shaped error curve in each dtype.
    name, idx, _ = picks[3]
    sweep = {}
    eps_values = [10.0**-k for k in np.arange(0, 10.5, 0.5)]
    print(f"== eps sweep for {name}{idx} ==")
    print(f"{'eps':>8} " + " ".join(f"{d:>10}" for d in DTYPES))
    for dname, dtype in DTYPES.items():
        model = copy.deepcopy(base).to(dtype)
        sweep[dname] = [check(model, x, y, [(name, idx, "")], e)[0]["rel_error"] for e in eps_values]
    for i, e in enumerate(eps_values):
        print(f"{e:>8.0e} " + " ".join(f"{sweep[d][i]:>10.1e}" for d in DTYPES))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checks": results, "sweep": {"weight": f"{name}{idx}", "eps": eps_values, "rel_error": sweep}}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
