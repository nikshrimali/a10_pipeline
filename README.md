# The honest loop

Train a small GPT on a real loop and make it report on itself: every tensor shape, a gradient checked by hand, gradient accumulation broken on purpose, grad norm watched for early warnings, and an MFU number we can defend.

## Setup

| | |
|---|---|
| Model | GPT written from scratch: 6 layers, width 384, 6 heads, context 256, tied embeddings, 12,297,984 params |
| Data | FineWeb-Edu, 25M train tokens, 0.5M held-out tokens, our own 4,096-token byte-level BPE (3.43 chars/token) |
| Hardware | MacBook Air M4 (8-core GPU, 16 GB, no fan), PyTorch 2.15 on MPS |
| Precision | bf16 autocast, fp32 master weights, gradients and AdamW state |
| Batch | 64 x 256 = 16,384 tokens/step, 1,519 steps (one pass over the data) |
| Optimizer | AdamW, lr 1e-3, 50 warmup steps, cosine to 1e-4, wd 0.1, clip 1.0 |
| Speed | fused AdamW + torch.compile, chosen by the MFU ladder |

Every step logs the four traces to `runs/main/metrics.jsonl`:

| trace | what it catches |
|---|---|
| loss | whether the model is learning at all |
| grad norm | trouble, before the loss shows it |
| tokens per second | whether something quietly got slower |
| MFU | whether the machine is being wasted |

MFU = 6 x N x tokens/s / peak. Apple publishes no peak, so we use 8 cores x 128 ALUs x 2 FLOPs x 1.6 GHz = 3.28 TFLOPS. The ALU count and clock speed come from community measurements, so this is an estimate. We also log the fraction of a measured 4096 x 4096 matmul ceiling.

## Results

All from `pipeline.ipynb`, which was run end to end on the M4 and kept its outputs.

**Training.** Held-out loss went from 8.396 to **3.925** (perplexity 51 out of 4,096 tokens) in 26.6 minutes of training steps.

**Gradient check (`gradcheck.py`).** In fp64, backward() and a central-difference nudge agree to 6.8 to 8.5 digits on 5 weights. In fp32 with eps = 1e-3 the nudge fails, with one wrong sign and one exact zero. Some nudges change the loss by only 3.5e-7, which is below fp32's resolution of about 1e-6 near a loss of 8.4.

**Broken accumulation (`accum.py`).** Micro-batches of 8, 32, 128 and 256 tokens. Averaging the averages gives an 8-token sequence's tokens 13.25x their fair weight. After 300 steps, held-out loss is **5.595 correct against 6.041 broken** (+0.45). At the same weights, the two gradients have cosine similarity 0.47.

**Grad norm before loss (`find_lead.py`).** The main run had no event, which is what a healthy run looks like. In a provoked run with the learning rate ramped from 1x to 20x over 60 steps, grad norm jumps at step 103 (z = 5.3, reaching z = 20 by step 107) and loss follows at step 110: **7 steps later**. A sudden x10 spike moves both in the same step.

**MFU.** Median 15,564 tok/s = 1.15 TFLOPS of useful work = **35.0% MFU**. That's 75% of the matmul ceiling measured at the start of the run.

**What costs the distance to 40%:**

1. **Heat, the biggest cost by far.** The same matmul ran at 2.81 TFLOPS at the start of the notebook and 1.53 TFLOPS after 12 minutes of load. On a cool chip, an earlier ladder run of the same code reached 57%.
2. **No fused attention on MPS.** PyTorch's attention kernel falls back to plain math in fp32 and stores the full (64, 6, 256, 256) score matrix: 604 MB, found by listing autograd's saved tensors.
3. **Unfused elementwise operations.** torch.compile's +20% (+34% when cool) shows what they cost.
4. **Small 384-wide matmuls.**
5. **The peak is an estimate.**

**The number 0.1 (`fp_bits.py`).** Worked by hand and checked bit for bit against PyTorch:

| format | stored value | error |
|---|---|---|
| fp32 | 0.100000001490116 | 0.0000015% |
| fp16 | 0.0999755859375 | 0.024% |
| bf16 | 0.10009765625 | 0.098% |
| fp8 E4M3 | 0.1015625 | 1.6% |
| fp8 E5M2 | 0.09375 | 6.3% |
| fp4 E2M1 | 0, below the smallest step of 0.5 | 100% |

Train in bf16 mixed precision. It has fp32's range, so no loss scaling is needed: 1e-8 survives and 70,000 doesn't overflow, while fp16 loses both. Keep fp32 master weights, because 1.0 + 0.001 rounds back to 1.0 in bf16.

## Findings we did not plan for

- **A torch.compile bug on MPS.** Compiled, the position-embedding gradient came out 64x (= batch size) too large, while the loss looked normal, because clipping and AdamW hide gradient scale. A minimal reproduction shows it only on MPS, only when the embedding is broadcast across the batch. `model.py` builds position ids as a (B, T) grid to avoid it, and `train.py --compile` now compares compiled and eager gradients before training (0.011% in the main run).
- **A 134 MB leak.** The Python variable `logits` kept each step's logits alive into the next step. Comparing the predicted memory bill with the device's count exposed it, and one `del` fixed it.

## Status

Complete on the M4. Two caveats in the saved notebook:

- Cells 4 and 9 ran before the logits fix, so their "before forward" memory reads 283 MB. Cell 10b ran on the fixed code and shows 149 MB.
- The ladder's absolute numbers were measured on a hot chip. A rerun will give different absolute numbers; the ranking of the rungs held in both runs.

Kaggle (T4) is supported but was not run.

## Files

| file | what |
|---|---|
| `guide.md` | the full walkthrough for beginners: concepts, number formats, how to run it, how to read every output |
| `pipeline.ipynb` | runs everything in order, with a markdown cell before every code cell; Mac or Kaggle |
| `data.py` | FineWeb-Edu download, BPE training, packed uint16 token files |
| `model.py` | the GPT, with shape tracing |
| `flops.py` | device, dtype, peak FLOPS, matmul ceiling, MFU, memory bill, saved-tensor listing |
| `train.py` | the loop and the four traces, `--trace-shapes`, `--compile` with a gradient check, `--lr-spike` / `--lr-ramp` |
| `gradcheck.py`, `accum.py`, `find_lead.py`, `ladder.py`, `fp_bits.py` | one experiment each |
| `plot.py` | `plots/trace.png`, `accum.png`, `lead.png`, `ladder.png`, `gradcheck.png` |

```bash
uv sync
uv run jupyter notebook pipeline.ipynb      # or run any script on its own; usage is at the top of each file
```
