"""Find a step where the grad norm moved before the loss did.

    uv run find_lead.py runs/main
    uv run find_lead.py runs/spike --skip 0

"Moved" needs a definition, or any wiggle counts. For each step we compare the value with the
previous --window steps: how many robust standard deviations is it above their median?

    z = (value - median of last W) / (1.4826 x median absolute deviation of last W)

(1.4826 x MAD is a standard deviation that one outlier cannot inflate.) A grad-norm event is a
step with z > --grad-z. A loss event is a step with z > --loss-z. We look for a grad-norm event
followed by a loss event within --horizon steps, with no loss event in the steps just before it,
so the grad norm really did move first.

Writes <run>/lead.json with every event found and the best one (largest grad-norm z).
"""

import argparse
import json

import numpy as np


def robust_z(x, window):
    """For each step t, how unusual x[t] is compared with x[t-window : t].

    Args:
        x: 1-D array of per-step values.
        window: how many previous steps form the baseline.

    Returns:
        Array of z-scores, NaN for the first `window` steps (no baseline yet).
    """
    z = np.full(len(x), np.nan)
    for t in range(window, len(x)):
        past = x[t - window : t]
        med = np.median(past)
        mad = np.median(np.abs(past - med)) * 1.4826
        z[t] = (x[t] - med) / max(mad, 1e-12)
    return z


def find_leads(steps, loss, gnorm, window, grad_z, loss_z, horizon, skip):
    """Return every (grad event, following loss event) pair in which the grad norm moved first.

    Args:
        steps, loss, gnorm: per-step arrays from metrics.jsonl.
        window: baseline length for robust_z.
        grad_z, loss_z: thresholds that count as "moved".
        horizon: how many steps after the grad event the loss has to react.
        skip: ignore the first `skip` steps (warmup, where everything moves at once).

    Returns:
        A list of dicts, one per event, with the steps, the lead in steps and both z-scores.
    """
    zg, zl = robust_z(gnorm, window), robust_z(loss, window)
    events = []
    t = max(window, skip)
    while t < len(steps):
        if zg[t] > grad_z and not np.any(zl[max(0, t - 3) : t + 1] > loss_z):
            after = [u for u in range(t + 1, min(len(steps), t + 1 + horizon)) if zl[u] > loss_z]
            if after:
                u = after[0]
                events.append({"grad_step": int(steps[t]), "loss_step": int(steps[u]), "lead": int(steps[u] - steps[t]),
                               "grad_z": float(zg[t]), "loss_z": float(zl[u]),
                               "grad_norm": float(gnorm[t]), "grad_norm_before": float(np.median(gnorm[t - window : t])),
                               "loss_at_event": float(loss[u]), "loss_before": float(np.median(loss[t - window : t]))})
                t = u + 1
                continue
        t += 1
    return events, zg, zl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--grad-z", type=float, default=4.0)
    ap.add_argument("--loss-z", type=float, default=4.0)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--skip", type=int, default=100, help="ignore warmup steps")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(f"{args.run}/metrics.jsonl")]
    recs = [r for r in recs if r["type"] == "train"]
    steps = np.array([r["step"] for r in recs])
    loss = np.array([r["loss"] for r in recs])
    gnorm = np.array([r["grad_norm"] for r in recs])
    events, zg, zl = find_leads(steps, loss, gnorm, args.window, args.grad_z, args.loss_z, args.horizon, args.skip)

    print(f"{len(recs)} steps; looking for grad-norm z > {args.grad_z} followed within {args.horizon} steps by "
          f"loss z > {args.loss_z} (baseline: previous {args.window} steps, first {args.skip} steps skipped)")
    if not events:
        print("no step where the grad norm moved first. Try the provoked run (train.py --lr-spike).")
    for e in events:
        print(f"  grad norm jumped at step {e['grad_step']} ({e['grad_norm_before']:.3f} -> {e['grad_norm']:.3f}, z={e['grad_z']:.1f}); "
              f"loss followed at step {e['loss_step']} ({e['loss_before']:.4f} -> {e['loss_at_event']:.4f}, z={e['loss_z']:.1f}); "
              f"lead {e['lead']} steps")
    best = max(events, key=lambda e: e["grad_z"]) if events else None
    if best:
        print(f"\nsteps around the clearest event (grad norm at {best['grad_step']}, loss at {best['loss_step']}):")
        print(f"  {'step':>5} {'loss':>8} {'loss z':>7} {'grad norm':>10} {'grad z':>7}")
        i0 = int(np.searchsorted(steps, best["grad_step"])) - 5
        i1 = int(np.searchsorted(steps, best["loss_step"])) + 6
        for i in range(max(0, i0), min(len(steps), i1)):
            mark = " <- grad norm moves" if steps[i] == best["grad_step"] else (" <- loss moves" if steps[i] == best["loss_step"] else "")
            print(f"  {steps[i]:>5} {loss[i]:>8.4f} {zl[i]:>7.1f} {gnorm[i]:>10.3f} {zg[i]:>7.1f}{mark}")
    with open(f"{args.run}/lead.json", "w") as f:
        json.dump({"settings": vars(args), "events": events, "best": best}, f, indent=2)
    print(f"wrote {args.run}/lead.json")


if __name__ == "__main__":
    main()
