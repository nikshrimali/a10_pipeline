"""Break gradient accumulation on purpose and measure the damage.

    uv run accum.py --data data/fwe --out runs/accum

Each optimizer step is built from K micro-batches whose sequences have different lengths
(8, 32, 128 and 256 tokens by default). There are two ways to combine their losses:

    correct   add up the loss of every token in all K micro-batches, divide by the total token count
    broken    take each micro-batch's mean loss, then average those K means ("average of averages")

With equal lengths the two are identical. With unequal lengths the broken version gives every
token in a short micro-batch far more weight than a token in a long one. This script trains the
same model twice from the same seed on the same data, once each way, and also measures, at every
step of the correct run, how far the broken gradient points away from the correct one.

Writes <out>/correct.jsonl and <out>/broken.jsonl. plot.py turns them into plots/accum.png.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch

import flops
from model import GPT, GPTConfig, token_loss
from train import configure_optimizer, evaluate, load_tokens, lr_at


def micro_batches(tokens, order, step, batch, lengths, dev):
    """Build the K micro-batches for one step, each with its own sequence length.

    Every micro-batch takes `batch` fresh windows from the shuffled order and keeps only the
    first L tokens of each, so micro-batch k holds batch * lengths[k] real tokens.

    Args:
        tokens: the uint16 train token array.
        order: shuffled window ids.
        step: optimizer step number (picks which windows to use, the same in both runs).
        batch: sequences per micro-batch.
        lengths: list of K sequence lengths.
        dev: device.

    Returns:
        A list of K (x, y) pairs, x and y of shape (batch, L_k).
    """
    w = max(lengths) + 1
    out = []
    for k, L in enumerate(lengths):
        start = (step * len(lengths) + k) * batch
        ids = order[start : start + batch]
        arr = np.stack([tokens[i * w : i * w + L + 1] for i in ids]).astype(np.int64)
        t = torch.from_numpy(arr).to(dev)
        out.append((t[:, :-1], t[:, 1:]))
    return out


def accumulate(model, mbs, mode, amp):
    """Run forward and backward over all micro-batches, accumulating gradients in .grad.

    Args:
        model: the GPT.
        mbs: output of micro_batches().
        mode: "correct" or "broken".
        amp: autocast context.

    Returns:
        (reported, true): the loss each method would print, and the true token-weighted mean.
        For "correct" they are the same number.

    Why it matters: backward() of (loss / something) scales every gradient by 1 / something.
    "correct" divides each micro-batch's summed loss by the step's total token count, so each
    token contributes 1 / total_tokens. "broken" divides each micro-batch's mean by K, so a token
    in micro-batch k contributes 1 / (K * tokens_k): short micro-batches get louder.
    """
    total_tokens = sum(y.numel() for _, y in mbs)
    K = len(mbs)
    reported, loss_sum = 0.0, 0.0
    for x, y in mbs:
        with amp:
            s = token_loss(model(x), y, reduction="sum")
        if mode == "correct":
            (s / total_tokens).backward()
            reported += s.item() / total_tokens
        else:
            mean = s / y.numel()
            (mean / K).backward()
            reported += mean.item() / K
        loss_sum += s.item()
    return reported, loss_sum / total_tokens


def flat_grad(model):
    """All gradients concatenated into one long fp32 vector."""
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters()])


def run(mode, args, dev, amp, tokens, eval_tokens, order, vocab):
    """Train one model for args.steps steps with the given accumulation mode and log every step.

    In the "correct" run we also compute the broken gradient at the same weights (an extra
    forward and backward that is thrown away) and log how different it is.
    """
    torch.manual_seed(args.seed)
    model = GPT(GPTConfig(vocab=vocab, context=max(args.lengths))).to(dev)
    opt = configure_optimizer(model, args.lr, 0.1)
    log = open(f"{args.out}/{mode}.jsonl", "w")
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps, 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        mbs = micro_batches(tokens, order, step, args.batch, args.lengths, dev)
        rec = {"step": step + 1}
        if mode == "correct":
            accumulate(model, mbs, "broken", amp)
            g_broken = flat_grad(model)
            opt.zero_grad(set_to_none=True)
        rec["reported_loss"], rec["true_loss"] = accumulate(model, mbs, mode, amp)
        if mode == "correct":
            g = flat_grad(model)
            rec["grad_rel_error"] = ((g_broken - g).norm() / g.norm()).item()
            rec["grad_cosine"] = torch.nn.functional.cosine_similarity(g_broken, g, dim=0).item()
        rec["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.eval_every == 0 or step == 0:
            rec["eval_loss"] = evaluate(model, eval_tokens, max(args.lengths), 32, amp, dev, args.eval_tokens)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if (step + 1) % 25 == 0 or step == 0:
            extra = f" grad error {rec['grad_rel_error']:.3f} cosine {rec['grad_cosine']:.4f}" if mode == "correct" else ""
            ev = f" eval {rec['eval_loss']:.4f}" if "eval_loss" in rec else ""
            print(f"[{mode}] step {step + 1}/{args.steps} reported {rec['reported_loss']:.4f} "
                  f"true {rec['true_loss']:.4f}{ev}{extra} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/accum")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8, 32, 128, 256])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-tokens", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = flops.device()
    amp_dtype = flops.autocast_dtype(dev)
    amp = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype != torch.float32 else contextlib.nullcontext()
    meta = json.load(open(f"{args.data}/meta.json"))
    tokens = load_tokens(args.data, "train")
    eval_tokens = load_tokens(args.data, "eval")
    order = np.random.default_rng(args.seed).permutation(len(tokens) // (max(args.lengths) + 1))

    K, B = len(args.lengths), args.batch
    total = B * sum(args.lengths)
    print(f"{K} micro-batches of {B} sequences, lengths {args.lengths}: {total:,} tokens per step")
    print("weight each single token gets in the step's loss:")
    print(f"  {'length':>6} {'tokens':>7} {'correct':>10} {'broken':>10} {'broken / correct':>17}")
    for L in args.lengths:
        c, b = 1 / total, 1 / (K * B * L)
        print(f"  {L:>6} {B * L:>7} {c:>10.2e} {b:>10.2e} {b / c:>16.2f}x")
    with open(f"{args.out}/config.json", "w") as f:
        json.dump({**vars(args), "tokens_per_step": total}, f, indent=2)

    for mode in ("correct", "broken"):
        run(mode, args, dev, amp, tokens, eval_tokens, order, meta["vocab_size"])

    rows = {m: [json.loads(l) for l in open(f"{args.out}/{m}.jsonl")] for m in ("correct", "broken")}
    ev = {m: [r["eval_loss"] for r in rows[m] if "eval_loss" in r] for m in rows}
    err = [r["grad_rel_error"] for r in rows["correct"]]
    print(f"\nfinal held-out loss: correct {ev['correct'][-1]:.4f}, broken {ev['broken'][-1]:.4f}, "
          f"gap {ev['broken'][-1] - ev['correct'][-1]:+.4f}")
    print(f"broken vs correct gradient, relative error: median {np.median(err):.3f}, "
          f"min {min(err):.3f}, max {max(err):.3f}")
    last = rows["broken"][-1]
    print(f"broken run's last step: it reports {last['reported_loss']:.4f}, the true token-weighted loss is "
          f"{last['true_loss']:.4f}")


if __name__ == "__main__":
    main()
