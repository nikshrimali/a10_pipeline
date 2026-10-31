"""The MFU ladder: add one speed trick at a time and measure what each one buys.

    uv run ladder.py --data data/fwe

Rungs, each adding to the one before:
    1  baseline        fp32 everywhere, attention written out by hand, 16 sequences per step
    2  + sdpa          PyTorch's built-in attention kernel instead of the hand-written one
    3  + bf16          autocast: matmuls in bf16, weights still fp32
    4  + batch 64      4x more sequences per step, so each kernel launch does more work
    5  + fused AdamW   the optimizer update as one kernel instead of one per parameter group
    6  + compile       torch.compile fuses small operations into fewer, bigger kernels

Each rung trains 10 steps and prints every one. Steps 1 to 3 include one-off costs (memory
allocation, kernel selection, compilation), so tokens per second and MFU come from steps 4 to 10.
At the end, the best working rung is timed phase by phase (forward, backward, optimizer) to show
where a step's time goes. Writes runs/ladder.json.
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch

import flops
from model import GPT, GPTConfig, count_params, token_loss
from train import configure_optimizer, get_batch, load_tokens

RUNGS = [
    ("baseline", dict(attn="manual", dtype="fp32", batch=16, fused=False, compile=False)),
    ("+ sdpa", dict(attn="sdpa", dtype="fp32", batch=16, fused=False, compile=False)),
    ("+ bf16 autocast", dict(attn="sdpa", dtype="bf16", batch=16, fused=False, compile=False)),
    ("+ batch 64", dict(attn="sdpa", dtype="bf16", batch=64, fused=False, compile=False)),
    ("+ fused AdamW", dict(attn="sdpa", dtype="bf16", batch=64, fused=True, compile=False)),
    ("+ torch.compile", dict(attn="sdpa", dtype="bf16", batch=64, fused=True, compile=True)),
]


def build(rung, vocab, dev, lr=1e-3):
    """Make a fresh model and optimizer for one rung.

    Args:
        rung: the settings dict from RUNGS.
        vocab: tokenizer size.
        dev: device.

    Returns:
        (model, forward function, optimizer, autocast context, compute dtype). The forward function
        is the compiled model when the rung asks for torch.compile, else the model itself.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab=vocab, attn=rung["attn"])).to(dev)
    opt = configure_optimizer(model, lr, 0.1, fused=rung["fused"])
    dtype = flops.autocast_dtype(dev, rung["dtype"])
    amp = torch.autocast(dev.type, dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()
    fwd = torch.compile(model) if rung["compile"] else model
    return model, fwd, opt, amp, dtype


def train_steps(model, fwd, opt, amp, tokens, order, batch, dev, n_steps, phases=False):
    """Run n_steps training steps and time each one with a device sync on both sides.

    Args:
        phases: if True, also sync between forward, backward and optimizer and time each part.
            The extra syncs slow the step slightly, so we only do this for the breakdown.

    Returns:
        A list of dicts with step time in seconds, and forward/backward/optimizer times if `phases`.
    """
    out = []
    for step in range(n_steps):
        # Ramp the learning rate over 10 steps. Speed does not depend on it, but starting at full
        # learning rate makes the loss jump around, which distracts from the timing.
        for g in opt.param_groups:
            g["lr"] = 1e-3 * min(1.0, (step + 1) / 10)
        x, y = get_batch(tokens, order[step * batch : (step + 1) * batch], 256, dev)
        flops.sync(dev)
        t0 = time.perf_counter()
        with amp:
            loss = token_loss(fwd(x), y)
        if phases:
            flops.sync(dev)
            t1 = time.perf_counter()
        loss.backward()
        if phases:
            flops.sync(dev)
            t2 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        flops.sync(dev)
        t3 = time.perf_counter()
        rec = {"step_time": t3 - t0, "loss": loss.item()}
        if phases:
            rec.update(forward=t1 - t0, backward=t2 - t1, optimizer=t3 - t2)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--timed-from", type=int, default=4, help="first step (1-based) counted in tok/s")
    ap.add_argument("--peak-tflops", type=float, default=None)
    ap.add_argument("--out", default="runs/ladder.json")
    args = ap.parse_args()

    dev = flops.device()
    meta = json.load(open(f"{args.data}/meta.json"))
    tokens = load_tokens(args.data, "train")
    order = np.random.default_rng(0).permutation(len(tokens) // 257)
    results = []
    best = None
    for name, rung in RUNGS:
        print(f"\n== {name}: attn={rung['attn']} dtype={rung['dtype']} batch={rung['batch']} "
              f"fused_adam={rung['fused']} compile={rung['compile']} ==", flush=True)
        try:
            model, fwd, opt, amp, dtype = build(rung, meta["vocab_size"], dev)
            n_params = count_params(model)
            peak, _ = flops.peak_flops(dev, dtype, args.peak_tflops)
            steps = train_steps(model, fwd, opt, amp, tokens, order, rung["batch"], dev, args.steps)
        except Exception as e:  # a rung that does not work on this machine is a result too
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            print(f"  FAILED: {msg}")
            results.append({"rung": name, **rung, "error": msg})
            continue
        tok_per_step = rung["batch"] * 256
        for i, s in enumerate(steps, 1):
            tps = tok_per_step / s["step_time"]
            m = flops.mfu(n_params, tps, peak)
            print(f"  step {i:2d}  {s['step_time'] * 1000:7.0f} ms  {tps:8,.0f} tok/s  "
                  f"mfu {100 * m:5.1f}%  loss {s['loss']:.3f}{'' if i >= args.timed_from else '   (warm-up, not counted)'}")
        timed = steps[args.timed_from - 1 :]
        tps = tok_per_step * len(timed) / sum(s["step_time"] for s in timed)
        m = flops.mfu(n_params, tps, peak)
        print(f"  -> steps {args.timed_from}-{args.steps}: {tps:,.0f} tok/s, MFU {100 * m:.1f}%")
        results.append({"rung": name, **rung, "tok_per_s": tps, "mfu": m,
                        "step_ms": [1000 * s["step_time"] for s in steps]})
        if best is None or tps > best[1]:
            best = (name, tps, rung)
        del model, fwd, opt
        if dev.type == "mps":
            torch.mps.empty_cache()

    print(f"\n== where the time goes, {best[0]} (5 steps after 3 warm-up steps, synced between phases) ==")
    model, fwd, opt, amp, dtype = build(best[2], meta["vocab_size"], dev)
    steps = train_steps(model, fwd, opt, amp, tokens, order, best[2]["batch"], dev, 8, phases=True)[3:]
    phase = {k: float(np.mean([s[k] for s in steps])) for k in ("forward", "backward", "optimizer", "step_time")}
    for k in ("forward", "backward", "optimizer"):
        print(f"  {k:<10} {1000 * phase[k]:7.1f} ms  {100 * phase[k] / phase['step_time']:5.1f}%")
    print(f"  {'total':<10} {1000 * phase['step_time']:7.1f} ms")

    print("\nsummary")
    print(f"  {'rung':<18} {'tok/s':>9} {'MFU':>7}")
    for r in results:
        if "error" in r:
            print(f"  {r['rung']:<18} {'failed':>9}")
        else:
            print(f"  {r['rung']:<18} {r['tok_per_s']:>9,.0f} {100 * r['mfu']:>6.1f}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rungs": results, "phases": {"rung": best[0], **phase}}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
