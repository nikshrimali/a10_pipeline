"""Train the small GPT and log the four traces at every step: loss, grad norm, tokens per second, MFU.

    uv run train.py --data data/fwe --out runs/main                       # the main run, ~30 min on an M4
    uv run train.py --data data/fwe --out runs/smoke --max-steps 30       # quick check
    uv run train.py --data data/fwe --out runs/shapes --max-steps 1 --trace-shapes
    uv run train.py --data data/fwe --out runs/spike --max-steps 150 --lr-spike 100:10:5
    uv run train.py --data data/fwe --out runs/ramp --max-steps 200 --lr-spike 100:20:60 --lr-ramp

Every step writes one line to <out>/metrics.jsonl. Steps are timed with a device sync on both
sides, and eval and checkpoint time happen outside the timed region, so tokens per second never
includes them. A held-out eval runs every --eval-every steps, a checkpoint is written every
--ckpt-minutes, and the run ends with a text sample.
"""

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from tokenizers import Tokenizer

import flops
from model import GPT, GPTConfig, count_params, token_loss

DIM_LEGEND = """Dimension letters:
  B  batch      sequences processed side by side in one micro-batch
  T  time       token positions in a sequence
  C  channels   numbers that represent one token (the model width)
  H  heads      independent attention patterns per layer
  D  head dim   channels each head works with, C / H
  V  vocab      possible tokens, one score each at the output
"""


def load_tokens(data_dir, split):
    """Open <data_dir>/<split>.bin as a read-only array of uint16 token ids without loading it into RAM."""
    return np.memmap(f"{data_dir}/{split}.bin", dtype=np.uint16, mode="r")


def get_batch(tokens, seq_ids, seq_len, dev):
    """Cut sequences out of the token array and make (input, target) pairs.

    Each sequence is seq_len + 1 consecutive tokens. The input is the first seq_len tokens and
    the target is the same window shifted one to the left, so target[t] is the token that comes
    after input[t]. That shift is the entire "predict the next token" task.

    Args:
        tokens: the uint16 token array.
        seq_ids: which windows to take; window i covers tokens[i*(seq_len+1) : (i+1)*(seq_len+1)].
        seq_len: T.
        dev: device to put the batch on.

    Returns:
        x: (B, T) int64 inputs, y: (B, T) int64 targets.
    """
    w = seq_len + 1
    batch = np.stack([tokens[i * w : (i + 1) * w] for i in seq_ids]).astype(np.int64)
    batch = torch.from_numpy(batch).to(dev)
    return batch[:, :-1], batch[:, 1:]


def configure_optimizer(model, lr, weight_decay, fused=False):
    """Build AdamW. Matrices get weight decay; LayerNorm gains and biases (1-D tensors) do not.

    Weight decay pulls every weight a little toward zero each step. It helps big matrices
    generalize, but on a LayerNorm gain it would just fight the normalization.

    Args:
        model: the GPT.
        lr: peak learning rate. The schedule scales it each step.
        weight_decay: decay strength for matrices.
        fused: use the single-kernel AdamW implementation (one of the ladder's rungs).

    Returns:
        A torch.optim.AdamW.
    """
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused or None)


def lr_at(step, peak, warmup, total, min_ratio):
    """Learning rate for this step: linear warmup to `peak`, then a cosine curve down to peak * min_ratio.

    Warmup avoids big, badly aimed steps while the early gradients are still noisy. The cosine
    tail takes smaller and smaller steps so the weights settle instead of bouncing around.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def spike_factor(step, spike, ramp=False):
    """Learning-rate multiplier for a deliberate disturbance. `spike` is (start, factor, length) or None.

    Without `ramp` the multiplier jumps straight to `factor` for `length` steps (a spike). With
    `ramp` it climbs linearly from 1 to `factor` over `length` steps, then drops back to 1. A
    spike hits loss and grad norm in the same step; a ramp lets instability build up gradually,
    which is when grad norm can move first.
    """
    if not spike or not spike[0] <= step < spike[0] + spike[2]:
        return 1.0
    if ramp:
        return 1.0 + (spike[1] - 1.0) * (step - spike[0] + 1) / spike[2]
    return spike[1]


@torch.no_grad()
def evaluate(model, tokens, seq_len, batch, amp, dev, max_tokens=None):
    """Average next-token loss on held-out text the model never trains on.

    Train loss can fall because the model memorizes its batches; held-out loss only falls if it
    learned something that carries over to new text.

    Args:
        model: the GPT.
        tokens: eval token array.
        seq_len, batch: window length and windows per forward pass.
        amp: the autocast context (or a null context for fp32).
        dev: device.
        max_tokens: stop after about this many tokens, None for all of them.

    Returns:
        Mean loss per token, as a float.
    """
    model.eval()
    n_seq = len(tokens) // (seq_len + 1)
    if max_tokens:
        n_seq = min(n_seq, max_tokens // seq_len)
    total, count = 0.0, 0
    for start in range(0, n_seq, batch):
        x, y = get_batch(tokens, range(start, min(start + batch, n_seq)), seq_len, dev)
        with amp:
            total += token_loss(model(x), y, reduction="sum").item()
        count += y.numel()
    model.train()
    return total / count


def write_shapes(path, model, opt, trace, loss, cfg, batch_shape):
    """Write shapes.txt: every tensor in one training step, with what each dimension means.

    Covers the forward activations (collected in `trace`), the loss, each parameter's gradient
    and AdamW's two state tensors. Blocks 1 to L-1 have the same shapes as block 0, so only
    block 0 is listed in full.
    """
    B, T = batch_shape
    lines = [f"One training step, B={B}, T={T}, C={cfg.width}, H={cfg.heads}, D={cfg.width // cfg.heads}, "
             f"V={cfg.vocab}, L={cfg.layers}", "", DIM_LEGEND, "== Forward ==", ""]
    for name, shape, dtype, meaning in trace:
        lines.append(f"{name:<26} {str(shape):<22} {dtype:<9} {meaning}")
    lines += ["", "== Loss ==", "",
              f"{'loss':<26} {str(tuple(loss.shape)):<22} {str(loss.dtype).replace('torch.', ''):<9} "
              "one number: the mean of -log(probability of the right token) over all B x T positions", "",
              "== Gradients (one per parameter, same shape and dtype as the parameter) ==", ""]
    for name, p in model.named_parameters():
        if name.startswith("blocks.") and not name.startswith("blocks.0."):
            continue
        lines.append(f"{name + '.grad':<34} {str(tuple(p.grad.shape)):<14} {str(p.grad.dtype).replace('torch.', ''):<9} "
                     f"{_param_meaning(name, cfg)}")
    lines.append(f"{'blocks.1-' + str(cfg.layers - 1) + '.*.grad':<34} same shapes as blocks.0")
    lines += ["", "== AdamW state (two per parameter: m = running mean of gradients, "
              "v = running mean of squared gradients) ==", ""]
    for name, p in model.named_parameters():
        if name.startswith("blocks.") and not name.startswith("blocks.0."):
            continue
        st = opt.state[p]
        lines.append(f"{name:<34} m {str(tuple(st['exp_avg'].shape)):<14} v {str(tuple(st['exp_avg_sq'].shape)):<14} "
                     f"{str(st['exp_avg'].dtype).replace('torch.', '')}")
    lines.append(f"{'blocks.1-' + str(cfg.layers - 1) + '.*':<34} same shapes as blocks.0")
    lines += ["", "Note: lm_head.weight is the same tensor as tok_emb.weight (weight tying), so it has one "
              "gradient and one pair of AdamW states, listed under tok_emb.weight."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _param_meaning(name, cfg):
    """One line describing a parameter by its name, for shapes.txt."""
    C, V = cfg.width, cfg.vocab
    table = {
        "tok_emb.weight": f"(V, C): one {C}-wide vector per token; also the output head (tied)",
        "pos_emb.weight": f"(context, C): one {C}-wide vector per position",
        "ln1.weight": "(C,): LayerNorm gain before attention", "ln1.bias": "(C,): LayerNorm shift before attention",
        "attn.qkv.weight": "(3C, C): makes queries, keys, values from each token",
        "attn.proj.weight": "(C, C): mixes the heads back together",
        "ln2.weight": "(C,): LayerNorm gain before the MLP", "ln2.bias": "(C,): LayerNorm shift before the MLP",
        "mlp.fc.weight": "(4C, C): widens each token to 4C", "mlp.proj.weight": "(C, 4C): narrows back to C",
        "ln_f.weight": "(C,): final LayerNorm gain", "ln_f.bias": "(C,): final LayerNorm shift",
    }
    for key, meaning in table.items():
        if name.endswith(key):
            return meaning
    return ""


def check_compiled_grads(model, fwd, x, y, amp, tol=0.01):
    """Compare the compiled model's gradients with plain PyTorch's on one batch, parameter by parameter.

    torch.compile rewrites the forward and backward passes. It should compute the same numbers,
    and on this project's first try it did not: on Apple GPUs the position-embedding gradient
    came out 64x too large (see guide Part 15). The loss looked fine, so only a check like this
    catches it.

    Args:
        model: the plain model. fwd: torch.compile(model).
        x, y: one batch. amp: the autocast context.
        tol: largest allowed relative difference in any parameter's gradient norm.

    Returns:
        The largest relative difference found. Raises SystemExit if it is above `tol`.
    """
    norms = []
    for f in (model, fwd):
        model.zero_grad(set_to_none=True)
        with amp:
            token_loss(f(x), y).backward()
        norms.append({n: p.grad.float().norm().item() for n, p in model.named_parameters()})
    model.zero_grad(set_to_none=True)
    worst_name, worst = max(((n, abs(norms[1][n] - norms[0][n]) / max(norms[0][n], 1e-12)) for n in norms[0]),
                            key=lambda kv: kv[1])
    print(f"compile check: largest gradient difference vs eager is {100 * worst:.3f}% ({worst_name})")
    if worst > tol:
        raise SystemExit(f"compiled gradients disagree with eager ones by {100 * worst:.1f}% on {worst_name}; "
                         "run without --compile")
    return worst


def print_memory_bill(n_params, cfg, batch_shape, amp_dtype, dev, measured, saved):
    """Print the textbook memory bill, then autograd's real saved-tensor list, then the device's own count.

    Args:
        measured: dict with bytes in use "before" the step (weights + optimizer state),
            "after_forward" (plus activations) and "after_backward" (plus gradients).
        saved: a flops.saved_tensors() that wrapped this step's forward pass.
    """
    rows = flops.memory_bill(n_params, cfg, batch_shape, amp_dtype)
    print("\nWhat one step costs in memory, textbook prediction:")
    for item, b, why in rows:
        print(f"  {item:<26} {b / 1e6:8.1f} MB   {why}")
    print(f"  {'total':<26} {sum(b for _, b, _ in rows) / 1e6:8.1f} MB")
    print(f"What autograd actually saved for backward: {saved.total_bytes / 1e6:.1f} MB")
    for shape, dtype, n, b in saved.by_kind():
        if b >= 1e6:
            print(f"  {b / 1e6:8.1f} MB  {n:3d} x {str(shape):<20} {dtype}")
    if dev.type != "cpu":
        print("Measured on the device (bytes held by tensors):")
        print(f"  before forward   {measured['before'] / 1e6:8.1f} MB   weights + AdamW state")
        print(f"  after forward    {measured['after_forward'] / 1e6:8.1f} MB   + activations saved for backward")
        print(f"  after backward   {measured['after_backward'] / 1e6:8.1f} MB   activations freed, gradients added")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--attn", choices=["sdpa", "manual"], default="sdpa")
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-tokens", type=float, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fused-adam", action="store_true")
    ap.add_argument("--peak-tflops", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-tokens", type=int, default=None, help="default: the whole eval split")
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--trace-shapes", action="store_true")
    ap.add_argument("--lr-spike", default=None, help="start:factor:length, e.g. 100:10:5 multiplies lr by 10 for 5 steps")
    ap.add_argument("--lr-ramp", action="store_true", help="make --lr-spike climb linearly from 1x to factor instead of jumping")
    ap.add_argument("--sample-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = flops.device()
    amp_dtype = flops.autocast_dtype(dev, args.dtype)
    amp = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype != torch.float32 else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(dev.type) if amp_dtype == torch.float16 else None
    spike = tuple(float(v) for v in args.lr_spike.split(":")) if args.lr_spike else None
    if spike:
        spike = (int(spike[0]), spike[1], int(spike[2]))

    meta = json.load(open(f"{args.data}/meta.json"))
    cfg = GPTConfig(vocab=meta["vocab_size"], context=args.seq_len, layers=args.layers, width=args.width,
                    heads=args.heads, attn=args.attn)
    model = GPT(cfg).to(dev)
    n_params = count_params(model)
    peak, peak_src = flops.peak_flops(dev, amp_dtype, args.peak_tflops)
    ceiling = flops.matmul_ceiling(dev, amp_dtype)
    print(f"model: {n_params:,} params ({n_params / 1e6:.2f}M), {flops.flops_per_token(n_params) / 1e6:.1f} MFLOPs per token (6N)")
    print(f"device: {dev}, matmuls in {amp_dtype}, loss scaling {'on' if scaler else 'off'}")
    print(f"peak: {peak_src}")
    print(f"measured matmul ceiling: {ceiling / 1e12:.2f} TFLOPS")

    train_tokens = load_tokens(args.data, "train")
    eval_tokens = load_tokens(args.data, "eval")
    n_seq = len(train_tokens) // (args.seq_len + 1)
    order = np.random.default_rng(args.seed).permutation(n_seq)
    seqs_per_step = args.micro_batch * args.grad_accum
    tokens_per_step = seqs_per_step * args.seq_len
    total_steps = n_seq // seqs_per_step
    if args.max_tokens:
        total_steps = min(total_steps, int(args.max_tokens) // tokens_per_step)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"{total_steps} steps x {tokens_per_step:,} tokens = {total_steps * tokens_per_step / 1e6:.1f}M tokens")

    opt = configure_optimizer(model, args.lr, args.weight_decay, args.fused_adam)
    step = 0
    ckpt_path = f"{args.out}/ckpt.pt"
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"resumed from {ckpt_path} at step {step}")
    fwd = torch.compile(model) if args.compile else model
    if args.compile:
        check_compiled_grads(model, fwd, *get_batch(train_tokens, order[: args.micro_batch], args.seq_len, dev), amp)

    log = open(f"{args.out}/metrics.jsonl", "a" if args.resume else "w")
    with open(f"{args.out}/config.json", "w") as f:
        json.dump({**vars(args), "model": asdict(cfg), "n_params": n_params, "peak_flops": peak,
                   "peak_source": peak_src, "matmul_ceiling": ceiling, "device": str(dev),
                   "amp_dtype": str(amp_dtype), "tokens_per_step": tokens_per_step, "total_steps": total_steps}, f, indent=2)

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg)}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    if step == 0 and args.eval_every:
        loss0 = evaluate(model, eval_tokens, args.seq_len, args.micro_batch, amp, dev, args.eval_tokens)
        write({"type": "eval", "step": 0, "tokens": 0, "loss": loss0})
        print(f"eval step 0: held-out loss {loss0:.4f} (random guessing would be ln({cfg.vocab}) = {math.log(cfg.vocab):.4f})")

    last_ckpt = time.time()
    window = []  # (tokens, seconds) of recent steps, for the printed tok/s
    flops.sync(dev)
    while step < total_steps:
        t0 = time.perf_counter()
        lr = lr_at(step, args.lr, args.warmup, total_steps, args.min_lr_ratio) * spike_factor(step, spike, args.lr_ramp)
        for g in opt.param_groups:
            g["lr"] = lr
        measure = step == 1  # by step 1 AdamW state exists, so the memory bill is complete
        mem = {"before": flops.memory_allocated(dev)} if measure else None
        step_loss_sum, step_tokens = 0.0, 0
        trace = [] if (args.trace_shapes and step == 0) else None
        for m in range(args.grad_accum):
            ids = order[(step * args.grad_accum + m) * args.micro_batch :][: args.micro_batch]
            x, y = get_batch(train_tokens, ids, args.seq_len, dev)
            with amp, (flops.saved_tensors() if measure else contextlib.nullcontext()) as saved:
                logits = model(x, trace=trace) if trace is not None else fwd(x)
                loss_sum = token_loss(logits, y, reduction="sum")
            if measure:
                flops.sync(dev)
                mem["after_forward"] = flops.memory_allocated(dev)
            # Divide by the token count of the whole step, not of this micro-batch (see accum.py).
            loss = loss_sum / (tokens_per_step)
            (scaler.scale(loss) if scaler else loss).backward()
            step_loss_sum += loss_sum.item()
            step_tokens += y.numel()
            # Without this, the Python names keep this step's logits (64 x 256 x 4096 bf16 = 134 MB)
            # alive until the next step overwrites them, so they sit on top of the next forward pass.
            del logits, loss_sum, loss
        if measure:
            flops.sync(dev)
            mem["after_backward"] = flops.memory_allocated(dev)
        if scaler:
            scaler.unscale_(opt)
        # clip_grad_norm_ returns the norm BEFORE clipping, which is the number we want to watch.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if trace is not None:
            write_shapes(f"{args.out}/shapes.txt", model, opt, trace, torch.tensor(step_loss_sum / step_tokens), cfg,
                         (args.micro_batch, args.seq_len))
        opt.zero_grad(set_to_none=True)
        flops.sync(dev)
        dt = time.perf_counter() - t0
        step += 1

        loss_val = step_loss_sum / step_tokens
        tps = step_tokens / dt
        rec = {"type": "train", "step": step, "tokens": step * tokens_per_step, "loss": loss_val,
               "grad_norm": grad_norm, "lr": lr, "step_time": dt, "tok_per_s": tps,
               "mfu": flops.mfu(n_params, tps, peak), "ceiling_frac": flops.mfu(n_params, tps, ceiling)}
        write(rec)
        if measure:
            print_memory_bill(n_params, cfg, (args.micro_batch, args.seq_len), amp_dtype, dev, mem, saved)
        window = (window + [(step_tokens, dt)])[-args.print_every:]
        if step <= 10 or step % args.print_every == 0 or step == total_steps:
            w_tps = sum(t for t, _ in window) / sum(s for _, s in window)
            w_mfu = flops.mfu(n_params, w_tps, peak)
            mfu_txt = f"{100 * w_mfu:.1f}%" if w_mfu is not None else "n/a"
            print(f"step {step}/{total_steps} loss {loss_val:.4f} grad_norm {grad_norm:.3f} lr {lr:.2e} "
                  f"{w_tps:,.0f} tok/s mfu {mfu_txt} ({dt * 1000:.0f} ms)", flush=True)

        # Everything below is outside the timed region.
        if args.eval_every and (step % args.eval_every == 0 or step == total_steps):
            ev = evaluate(model, eval_tokens, args.seq_len, args.micro_batch, amp, dev, args.eval_tokens)
            write({"type": "eval", "step": step, "tokens": step * tokens_per_step, "loss": ev})
            print(f"eval step {step}: held-out loss {ev:.4f}", flush=True)
        if time.time() - last_ckpt > args.ckpt_minutes * 60 and step < total_steps:
            save()
            last_ckpt = time.time()
            print(f"saved {ckpt_path} at step {step}", flush=True)

    save()
    print(f"saved {ckpt_path}")
    if args.sample_tokens and os.path.exists(f"{args.data}/tokenizer.json"):
        tok = Tokenizer.from_file(f"{args.data}/tokenizer.json")
        prompt = "The most important thing about learning is"
        torch.manual_seed(0)
        with amp:
            out = model.generate(torch.tensor([tok.encode(prompt).ids], device=dev), args.sample_tokens)
        text = tok.decode(out[0].tolist())
        print(f"\nsample:\n{text}")
        with open(f"{args.out}/sample.txt", "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
