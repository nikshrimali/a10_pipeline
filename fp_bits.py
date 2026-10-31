"""Write out 0.1 bit by bit in six floating-point formats, then check every bit against PyTorch.

    uv run fp_bits.py
    uv run fp_bits.py --value 3.14159

Every format here stores a number as   (-1)^sign  x  1.mantissa  x  2^(exponent - bias).
They differ only in how many bits go to the exponent (range: how big and how small) and how
many go to the mantissa (precision: how many digits survive).

    format     sign  exponent  mantissa  bias
    fp32         1       8        23      127
    fp16         1       5        10       15
    bf16         1       8         7      127
    fp8 E4M3     1       4         3        7
    fp8 E5M2     1       5         2       15
    fp4 E2M1     1       2         1        1

The encoder below does the rounding with exact fractions, the same way the hardware does
(round to nearest, ties to even), and prints each step so you can follow it by hand.
"""

import argparse
from fractions import Fraction

import torch

# name: (exponent bits, mantissa bits, bias, largest finite value, torch dtype or None)
FORMATS = {
    "fp32": (8, 23, 127, None, torch.float32),
    "fp16": (5, 10, 15, None, torch.float16),
    "bf16": (8, 7, 127, None, torch.bfloat16),
    "fp8 E4M3": (4, 3, 7, Fraction(448), torch.float8_e4m3fn),
    "fp8 E5M2": (5, 2, 15, None, torch.float8_e5m2),
    "fp4 E2M1": (2, 1, 1, Fraction(6), None),
}


def binary_fraction(x, n):
    """The first n binary digits after the point of a number between 0 and 1.

    Uses the doubling method you can do on paper: double the number; if it reaches 1, write 1
    and subtract 1, otherwise write 0. Repeat.

    Args:
        x: a Fraction in [0, 1).
        n: how many digits to produce.

    Returns:
        A string of '0' and '1' characters.
    """
    bits = []
    for _ in range(n):
        x *= 2
        if x >= 1:
            bits.append("1")
            x -= 1
        else:
            bits.append("0")
    return "".join(bits)


def max_finite(e_bits, m_bits, bias, override):
    """Largest finite value of a format. IEEE formats reserve the all-ones exponent for inf and NaN."""
    if override is not None:
        return override
    return (2 - Fraction(1, 2**m_bits)) * Fraction(2) ** ((2**e_bits - 2) - bias)


def encode(value, name):
    """Round `value` into format `name`, exactly, and explain every step.

    Args:
        value: a positive Fraction (or something Fraction() accepts).
        name: a key of FORMATS.

    Returns:
        A dict with the sign, exponent and mantissa bit strings, the stored value as a Fraction,
        and `steps`, a list of lines describing the derivation in words.
    """
    e_bits, m_bits, bias, override, _ = FORMATS[name]
    v = Fraction(value)
    steps = []
    vmax = max_finite(e_bits, m_bits, bias, override)
    if v > vmax:
        return {"name": name, "overflow": True, "stored": None,
                "steps": [f"{float(v):g} is above the largest {name} value {float(vmax):g}: overflow (inf, or NaN in E4M3)"]}
    # Find e with 1 <= v / 2^e < 2. That e is the "true" exponent.
    e = 0
    while v / Fraction(2) ** e >= 2:
        e += 1
    while v / Fraction(2) ** e < 1:
        e -= 1
    e_min = 1 - bias  # smallest exponent a normal number can have
    if e >= e_min:
        sig = v / Fraction(2) ** e
        steps.append(f"write it as 1.something x 2^e: {float(v):g} = {float(sig):.10g} x 2^{e}")
        frac = sig - 1
        steps.append(f"mantissa = the part after '1.' in binary: {float(frac):.10g} -> 0.{binary_fraction(frac, m_bits + 8)}...")
        scaled = frac * 2**m_bits
        exp_field = e + bias
        steps.append(f"stored exponent = {e} + bias {bias} = {exp_field}")
    else:
        scaled = v / Fraction(2) ** e_min * 2**m_bits
        exp_field = 0
        steps.append(f"{float(v):g} is below the smallest normal number 2^{e_min} = {2.0 ** e_min:g}, so it becomes "
                     f"subnormal: exponent field 0, value = 0.mantissa x 2^{e_min}")
    keep = int(scaled)  # floor, since scaled >= 0
    rest = scaled - keep
    kept = format(keep, f"0{m_bits}b") if m_bits else ""
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and keep % 2 == 1):
        keep += 1
        how = "the dropped bits are more than half a step (or exactly half and the last bit is odd): round UP"
    else:
        how = "the dropped bits are less than half a step: round DOWN (just cut them off)"
    steps.append(f"keep {m_bits} mantissa bits: {kept} | dropped part = {float(rest):.4f} of one last-bit step; {how}")
    if keep == 2**m_bits:  # rounding carried into the exponent, e.g. 1.111 + 0.001 = 10.000
        keep = 0
        exp_field += 1
        steps.append("rounding carried over: mantissa resets to all zeros, exponent goes up by one")
    if exp_field == 0:
        stored = Fraction(keep, 2**m_bits) * Fraction(2) ** e_min
    else:
        stored = (1 + Fraction(keep, 2**m_bits)) * Fraction(2) ** (exp_field - bias)
    if stored > vmax:
        return {"name": name, "overflow": True, "stored": None, "steps": steps + ["rounded past the largest value: overflow"]}
    out = {"name": name, "overflow": False, "sign": "0", "exponent": format(exp_field, f"0{e_bits}b"),
           "mantissa": format(keep, f"0{m_bits}b") if m_bits else "", "stored": stored, "steps": steps}
    out["steps"].append(f"bits: {out['sign']} | {out['exponent']} | {out['mantissa']}  ->  stored value {float(stored):.12g}")
    return out


def torch_bits(value, name):
    """Ask PyTorch to convert `value` and return (sign, exponent, mantissa) bit strings, or None if unsupported."""
    e_bits, m_bits, _, _, dtype = FORMATS[name]
    if dtype is None:
        return None
    t = torch.tensor(float(value), dtype=torch.float64).to(dtype)
    width = 1 + e_bits + m_bits
    int_dtype = {32: torch.int32, 16: torch.int16, 8: torch.uint8}[width]
    raw = int(t.view(int_dtype).item()) & ((1 << width) - 1)
    b = format(raw, f"0{width}b")
    return b[0], b[1 : 1 + e_bits], b[1 + e_bits :], float(t.to(torch.float64).item())


def fp4_table():
    """All 16 fp4 E2M1 codes and their values, decoded with the formula by hand.

    PyTorch only has a packed two-per-byte fp4 type without a conversion we can check against,
    so for fp4 we list every code instead and confirm the nearest one matches encode().
    """
    rows = []
    for code in range(16):
        s, e, m = code >> 3, (code >> 1) & 3, code & 1
        val = (m / 2) * 2.0**0 if e == 0 else (1 + m / 2) * 2.0 ** (e - 1)
        rows.append((format(code, "04b"), -val if s else val))
    return rows


def describe_format(name):
    """One line with the format's range and precision: largest value, smallest normal, decimal digits."""
    e_bits, m_bits, bias, override, _ = FORMATS[name]
    vmax = float(max_finite(e_bits, m_bits, bias, override))
    min_normal = 2.0 ** (1 - bias)
    min_sub = 2.0 ** (1 - bias - m_bits)
    digits = (m_bits + 1) * 0.30103
    return (f"{name:<9} max {vmax:<11.4g} smallest normal {min_normal:<10.3g} smallest subnormal {min_sub:<10.3g} "
            f"~{digits:.1f} decimal digits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="0.1", help="positive number to encode, e.g. 0.1 or 1/3")
    args = ap.parse_args()
    value = Fraction(args.value)

    print("Range and precision of each format:")
    for name in FORMATS:
        print("  " + describe_format(name))

    print(f"\n{float(value):g} in binary is 0.{binary_fraction(value % 1, 40)}... (it never ends for 0.1)\n")
    for name in FORMATS:
        r = encode(value, name)
        print(f"== {name} ==")
        for line in r["steps"]:
            print("  " + line)
        if r["stored"] is not None:
            err = abs(float(r["stored"]) - float(value)) / float(value)
            print(f"  error: {float(r['stored']) - float(value):+.3e} ({100 * err:.4f}% of the value)")
        tb = torch_bits(value, name)
        if tb:
            ok = r["stored"] is not None and (tb[0], tb[1], tb[2]) == (r["sign"], r["exponent"], r["mantissa"])
            print(f"  torch: {tb[0]} | {tb[1]} | {tb[2]}  -> {tb[3]:.12g}   {'MATCH' if ok else 'MISMATCH'}")
        elif name == "fp4 E2M1":
            table = fp4_table()
            print("  all 16 fp4 codes: " + ", ".join(f"{c}={v:g}" for c, v in table[:8]) + " (and their negatives)")
            nearest = min(table[:8], key=lambda cv: abs(cv[1] - float(value)))
            ok = r["stored"] is not None and nearest[0] == "0" + r["exponent"] + r["mantissa"]
            print(f"  nearest code by brute force: {nearest[0]} = {nearest[1]:g}   {'MATCH' if ok else 'MISMATCH'}")
        print()

    print("Why bf16 beats fp16 for training, in two numbers:")
    for x, why in ((Fraction(1, 10**8), "a tiny gradient"), (Fraction(70000), "a big activation or loss-scaled value")):
        for name in ("fp16", "bf16"):
            r = encode(x, name)
            if r["overflow"]:
                shown = "overflow -> inf"
            elif r["stored"] == 0:
                shown = "0 (underflow: the gradient is lost)"
            else:
                shown = f"{float(r['stored']):.6g}"
            print(f"  {why:<38} {float(x):<8g} in {name}: {shown}")


if __name__ == "__main__":
    main()
