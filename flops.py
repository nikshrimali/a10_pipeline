"""The machine side of the run: which device, which precision, how fast it could go, what memory a step needs.

    import flops
    dev = flops.device()
    amp_dtype = flops.autocast_dtype(dev)
    peak, source = flops.peak_flops(dev, amp_dtype)
    mfu = flops.mfu(n_params, tokens_per_second, peak)

MFU (model FLOPs utilization) answers "what fraction of the machine are we using?":

    MFU = 6 x N x tokens per second / what the machine can do per second

6N is the cost of one token through forward and backward for a model with N weights: each
weight takes part in one multiply and one add going forward (2 FLOPs), and twice that going
backward (4 FLOPs), once for the gradient of the input and once for the gradient of the weight.
"""

import re
import subprocess
import time

import torch

# Dense (no sparsity) peak FLOPS from each vendor's datasheet, in TFLOPS, per compute dtype.
# Missing dtype means the chip has no fast path for it (the T4 emulates bf16).
CUDA_PEAKS = {
    "T4": {"fp16": 65.0, "fp32": 8.1},
    "P100": {"fp16": 18.7, "fp32": 9.3},
    "V100": {"fp16": 125.0, "fp32": 15.7},
    "L4": {"bf16": 121.0, "fp16": 121.0, "fp32": 30.3},
    "A100": {"bf16": 312.0, "fp16": 312.0, "fp32": 19.5},
    "H100": {"bf16": 989.0, "fp16": 989.0, "fp32": 67.0},
}

# Apple publishes no FLOPS figure. Peak = GPU cores x 128 FP32 ALUs per core x 2 FLOPs per fused
# multiply-add x clock. The ALU count and the clock come from community measurements of the chip,
# not from Apple, so treat the result as an estimate. Apple GPUs run fp16/bf16 at the same rate
# as fp32 (no tensor cores), so one number covers every dtype.
APPLE_ALUS_PER_CORE = 128
APPLE_CLOCK_GHZ = {"M4": 1.6}


def device():
    """Pick the fastest device available: an NVIDIA GPU, then an Apple GPU (MPS), then the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(dev, choice="auto"):
    """Choose the dtype matmuls run in. The weights themselves always stay fp32.

    Args:
        dev: the torch device.
        choice: "auto", "bf16", "fp16" or "fp32".

    Returns:
        torch.bfloat16 on Apple GPUs and NVIDIA Ampere or newer, torch.float16 on older NVIDIA
        GPUs such as the T4 (bf16 works there but is emulated and much slower), torch.float32 on CPU.
    """
    if choice != "auto":
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[choice]
    if dev.type == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    if dev.type == "mps":
        return torch.bfloat16
    return torch.float32


def sync(dev):
    """Wait until the GPU has finished all queued work.

    GPU calls return immediately and run later. Without a sync, a timer measures how fast Python
    queued the work, not how fast the GPU did it.
    """
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def memory_allocated(dev):
    """Bytes currently held by tensors on the device (0 on CPU, where we do not track it)."""
    if dev.type == "cuda":
        return torch.cuda.memory_allocated()
    if dev.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def _dtype_name(dtype):
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]


def apple_gpu():
    """Return (chip name, GPU core count) for this Mac, e.g. ("Apple M4", 8), or (None, None)."""
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        info = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
        cores = int(re.search(r"Total Number of Cores:\s*(\d+)", info).group(1))
        return chip, cores
    except (OSError, AttributeError, ValueError):
        return None, None


def peak_flops(dev, dtype, override_tflops=None):
    """The denominator of MFU: the most FLOPs per second this device can do in `dtype`.

    Args:
        dev: the torch device.
        dtype: the dtype matmuls run in (the autocast dtype).
        override_tflops: use this number instead of looking it up.

    Returns:
        (flops_per_second or None, a sentence saying where the number came from).
    """
    if override_tflops:
        return override_tflops * 1e12, f"{override_tflops} TFLOPS from --peak-tflops"
    name = _dtype_name(dtype)
    if dev.type == "cuda":
        gpu = torch.cuda.get_device_name(dev)
        for key, peaks in CUDA_PEAKS.items():
            if key in gpu:
                if name in peaks:
                    return peaks[name] * 1e12, f"{gpu} datasheet, dense {name}: {peaks[name]} TFLOPS"
                return peaks["fp32"] * 1e12, f"{gpu} has no fast {name}; using its fp32 peak {peaks['fp32']} TFLOPS"
        return None, f"{gpu} is not in CUDA_PEAKS, pass --peak-tflops"
    if dev.type == "mps":
        chip, cores = apple_gpu()
        family = next((k for k in APPLE_CLOCK_GHZ if chip and k in chip), None)
        if family is None or cores is None:
            return None, f"no clock estimate for {chip}, pass --peak-tflops"
        ghz = APPLE_CLOCK_GHZ[family]
        peak = cores * APPLE_ALUS_PER_CORE * 2 * ghz * 1e9
        return peak, (f"{chip}, {cores} GPU cores x {APPLE_ALUS_PER_CORE} ALUs x 2 FLOPs x {ghz} GHz "
                      f"= {peak / 1e12:.2f} TFLOPS (estimate, same for every dtype)")
    return None, "no peak for CPU, pass --peak-tflops"


def matmul_ceiling(dev, dtype, n=4096, iters=20):
    """Measure how fast one big square matmul runs, the most a real kernel gets out of this chip.

    A 4096 x 4096 matmul does 2 * n^3 FLOPs (n^3 multiplies and n^3 adds) and is about as
    friendly a workload as a GPU ever sees, so it is a practical ceiling for MFU.

    Returns:
        Achieved FLOPs per second.
    """
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(3):
        a @ b
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync(dev)
    return 2 * n**3 * iters / (time.perf_counter() - t0)


def flops_per_token(n_params):
    """6N: forward costs 2 FLOPs per weight per token, backward costs 4."""
    return 6 * n_params


def mfu(n_params, tokens_per_second, peak):
    """MFU = 6 x N x tokens per second / peak FLOPs per second. Returns None when the peak is unknown."""
    if not peak:
        return None
    return flops_per_token(n_params) * tokens_per_second / peak


def memory_bill(n_params, cfg, batch, amp_dtype):
    """Predict, from the textbook, what one training step holds in memory.

    "Textbook" means: assume attention runs as one fused kernel (flash attention) that never
    stores the (T, T) score matrix, which is what you get on a recent NVIDIA GPU. saved_tensors()
    then measures what autograd really keeps on this machine, and the two are printed side by side.

    Args:
        n_params: N.
        cfg: the GPTConfig (for layers L, width C, heads H, vocab V).
        batch: (B, T), sequences per micro-batch and tokens per sequence.
        amp_dtype: dtype of activations under autocast.

    Returns:
        A list of (item, bytes, explanation) rows.
    """
    B, T = batch
    L, C, H, V = cfg.layers, cfg.width, cfg.heads, cfg.vocab
    a = torch.finfo(amp_dtype).bits // 8
    matmul_params = n_params - L * 4 * C - 2 * C - cfg.context * C  # LayerNorms and pos_emb are not cast
    # Per block: two fp32 copies of the residual stream (inputs to the two LayerNorms), and in the
    # autocast dtype: ln1 out, q, k, v, attention out, ln2 out, MLP hidden before and after GELU.
    per_block = 2 * 4 * B * T * C + a * B * T * C * (1 + 3 + 1 + 1 + 4 + 4) + 4 * B * H * T
    rows = [
        ("weights (fp32)", 4 * n_params, f"{n_params:,} params x 4 bytes"),
        ("gradients (fp32)", 4 * n_params, "one gradient per weight, same dtype as the weight"),
        ("AdamW state (fp32)", 8 * n_params, "two running averages per weight (m and v), 4 bytes each"),
    ]
    if a < 4:
        rows.append((f"{_dtype_name(amp_dtype)} weight copies", a * matmul_params,
                     "autocast casts each matmul weight once per forward and keeps the copy for backward"))
    rows += [
        ("activations, blocks", L * per_block, f"{L} blocks x tensors saved for backward at B={B}, T={T}"),
        ("activations, loss", B * T * V * 4, "fp32 log-probabilities that cross-entropy keeps for backward"),
    ]
    return rows


class saved_tensors:
    """Context manager that records every tensor autograd saves for the backward pass.

    Backward needs some forward values (the input of every matmul, the output of every softmax...),
    and autograd keeps them alive until backward runs. That is the "activation memory". Wrapping a
    forward pass in this context lists them exactly, instead of guessing.

        with flops.saved_tensors() as saved:
            loss = token_loss(model(x), y)
        saved.total_bytes, saved.by_kind()

    Tensors that share storage (views of one buffer) are counted once.
    """

    def __enter__(self):
        self.seen = {}
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        self._hooks.__exit__(*exc)

    def _pack(self, t):
        key = t.untyped_storage().data_ptr()
        if key not in self.seen and t.device.type != "cpu":
            self.seen[key] = (tuple(t.shape), str(t.dtype).replace("torch.", ""), t.untyped_storage().nbytes())
        return t

    @property
    def total_bytes(self):
        """Bytes of all unique saved tensors."""
        return sum(b for _, _, b in self.seen.values())

    def by_kind(self):
        """Group saved tensors by (shape, dtype). Returns [(shape, dtype, count, bytes)], biggest first."""
        groups = {}
        for shape, dtype, b in self.seen.values():
            n, tot = groups.get((shape, dtype), (0, 0))
            groups[(shape, dtype)] = (n + 1, tot + b)
        return sorted(((s, d, n, b) for (s, d), (n, b) in groups.items()), key=lambda r: -r[3])
