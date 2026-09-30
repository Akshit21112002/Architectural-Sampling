"""
Best-of-n (K / block-window) eval on RealWorldQA. Model-agnostic: the model is
chosen by which RVD backend you import below (Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL).

RealWorldQA (xai-org/RealworldQA): 765 real-world images, each with a question
and a short, verifiable answer. Multiple-choice options are embedded in the
question text (no separate 'choices' column); MCQ answers are a letter, others
are a short word/number.

Scoring is BINARY correctness via a rule-based verifier:
  - if GT is a single letter (A..H): parse the choices out of the question,
    accept either the predicted letter OR the predicted option text (mapped
    back to its letter)
  - if GT is numeric: numeric match (integer/near-equal)
  - else: normalized string match (with a guarded substring fallback)

The model is told to answer, then put the final answer in \boxed{...}.
A sample is "solved" the first time any variation is correct, then dropped.
Best-of-n = cumulative solved / N. Temp baseline = unbiased pass@k.

NOTE: this is a deterministic offline verifier (no LLM judge). Absolute accuracy
may differ slightly from leaderboards, but the search-vs-temp comparison is
apples-to-apples since both use the same verifier.

Usage:
    python realworldqa_bon.py \
        --model /Qwen2.5-VL-7B-Instruct \
        --dataset_dir /data/RealWorldQA \
        --max_samples 765 \
        --K_eval 1 2 4 --block_min 0 --block_max 5 --block_window 3 \
        --max_new_tokens 256 \
        --output realworldqa_bon_results.txt \
        --plot_output realworldqa_bon_curve.png
"""

import argparse
import io
import re
import string
from math import comb
from pathlib import Path

import torch
from datasets import load_from_disk, Dataset, DatasetDict
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# RVD backend — pick the one that matches your model, uncomment exactly ONE.
#   Qwen2.5-VL / Qwen3-VL  -> transformers 4.57.0  (the default environment)
#   Qwen3.5-VL             -> transformers 5.16.1  (pip install -U transformers)
# -----------------------------------------------------------------------------
# import rvd_qwen_2_5_evo as rvd     # Qwen2.5-VL
import rvd_qwen_3_evo   as rvd   # Qwen3-VL
# import rvd_qwen_3_5_evo as rvd   # Qwen3.5-VL  (needs transformers 5.16.1)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default="/models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora_dir", default=None)
    p.add_argument("--dataset_dir", required=True, type=str)
    p.add_argument("--dataset_split", default="test")
    p.add_argument("--max_samples", type=int, default=765)

    p.add_argument("--K_eval", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--block_min", type=int, default=0)
    p.add_argument("--block_max", type=int, default=21)
    p.add_argument("--block_window", type=int, default=3)

    p.add_argument("--cot", action="store_true",
                   help="Ask the model to reason step by step before the boxed answer "
                        "(off by default; RealWorldQA expects short direct answers).")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=str, default="realworldqa_bon_results.txt")

    p.add_argument("--plot_output", type=str, default="realworldqa_bon_curve.png")
    p.add_argument("--skip_temp_bon", action="store_true")
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--temp_bon_samples", type=int, default=None)
    p.add_argument("--temp_bon_batch", type=int, default=4)
    p.add_argument("--temp_bon_estimator", choices=["unbiased", "prefix"], default="prefix")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_ds(dataset_dir, split):
    ds = load_from_disk(dataset_dir)
    if isinstance(ds, DatasetDict) or (not isinstance(ds, Dataset) and hasattr(ds, "keys")):
        keys = list(ds.keys())
        if split is None or split not in keys:
            split = "test" if "test" in keys else keys[0]
        print(f"[data] DatasetDict found, using split='{split}' (available: {keys})")
        ds = ds[split]
    return ds


def _get(ex, names, default=None):
    for k in names:
        if k in ex and ex[k] is not None:
            return ex[k]
    return default


def get_image(ex):
    img = _get(ex, ["image", "decoded_image", "Image", "img"])
    if isinstance(img, list):
        img = img[0]
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, str) and Path(img).exists():
        return Image.open(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


# -----------------------------------------------------------------------------
# Prompt + verifier
# -----------------------------------------------------------------------------

_LETTERS = "ABCDEFGH"
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# Choice markers embedded in the question, e.g. "A.", "(B)", "C)" — require a
# non-letter just before so we don't match inside words/abbreviations.
_MARK_RE = re.compile(r"(?<![A-Za-z])\(?([A-H])[\.\):]")


def parse_choices(question):
    """Return {letter: normalized_option_text} parsed from the question text.
    Splits on choice markers and stops each option at ';' or newline."""
    marks = [(m.start(), m.group(1).upper(), m.end()) for m in _MARK_RE.finditer(question)]
    out = {}
    for i, (_, letter, end) in enumerate(marks):
        text_end = marks[i + 1][0] if i + 1 < len(marks) else len(question)
        txt = re.split(r"[;\n]", question[end:text_end])[0]
        out[letter] = _norm_text(txt)
    return out


def build_prompt(question, cot):
    instr = ("Reason briefly, then give ONLY your final answer inside \\boxed{}."
             if cot else
             "Give ONLY your final answer (a letter, word, or number) inside \\boxed{}.")
    return f"{question.strip()}\n{instr}"


def extract_answer(text):
    if not text:
        return ""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*(.+)", text, re.I)
    if m:
        tail = m.group(1).strip().splitlines()
        if tail and tail[0].strip():
            return tail[0].strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def _norm_text(s):
    s = str(s).strip().lower().replace("%", "").replace("$", "")
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _to_num(s):
    m = _NUM_RE.search(str(s).replace(",", ""))
    return float(m.group().replace(",", "")) if m else None


def _pred_letter(pred, choice_map):
    """Get a choice letter from the prediction: a standalone letter, else by
    matching the predicted text to a parsed option."""
    for L in _LETTERS:
        if re.search(rf"(?:^|[^A-Za-z])\(?{L}\)?(?:[^A-Za-z]|$)", pred):
            if not choice_map or L in choice_map:
                return L
    np_ = _norm_text(pred)
    for L, txt in choice_map.items():
        if np_ == txt or (len(txt) >= 3 and txt in np_):
            return L
    return None


def verify(pred_raw, gt, question):
    pred = extract_answer(pred_raw)
    g = str(gt).strip()

    # --- multiple choice (GT is a single letter)
    if len(g) == 1 and g.upper() in _LETTERS:
        choice_map = parse_choices(question)
        pl = _pred_letter(pred, choice_map)
        return int(pl is not None and pl == g.upper())

    # --- numeric
    gn = _to_num(g)
    if gn is not None:
        pn = _to_num(pred)
        if pn is None:
            return 0
        if float(gn).is_integer():
            return int(round(pn) == round(gn))
        return int(abs(pn - gn) <= 0.5 * 10 ** (-2) or round(pn, 2) == round(gn, 2))

    # --- text
    np_, ng = _norm_text(pred), _norm_text(g)
    if np_ == ng:
        return 1
    if len(ng) >= 3 and (re.search(rf"\b{re.escape(ng)}\b", np_) or
                         re.search(rf"\b{re.escape(np_)}\b", ng)):
        return 1
    return 0


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def _build_inputs(processor, image, prompt, device):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        # Qwen3.5-VL only: add `enable_thinking=False,` here to disable thinking mode.
    )
    return processor(text=[text], images=[image],
                     max_pixels=512 * 28 * 28, return_tensors="pt").to(device)


@torch.no_grad()
def run_one(model, processor, image, prompt, max_new_tokens, device):
    inputs = _build_inputs(processor, image, prompt, device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


@torch.no_grad()
def run_samples(model, processor, image, prompt, max_new_tokens, device,
                n_samples, temperature, batch_size):
    inputs = _build_inputs(processor, image, prompt, device)
    preds, remaining = [], n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=temperature,
                             num_return_sequences=b)
        gen = out[:, inputs["input_ids"].shape[1]:]
        preds.extend(processor.tokenizer.batch_decode(gen, skip_special_tokens=True))
        remaining -= b
    return preds


# -----------------------------------------------------------------------------
# Cascade helpers
# -----------------------------------------------------------------------------

def build_variations(K_eval, block_min, block_max, window):
    if window <= 0:
        raise ValueError("--block_window must be >= 1")
    if block_max < block_min + window - 1:
        raise ValueError(f"block range too small for window={window}")
    variations = []
    for K in K_eval:
        if K == 1:
            variations.append({"K": 1, "block_start": 1, "block_end": 3})
            continue
        for bs in range(block_min, block_max - window + 2):
            variations.append({"K": int(K), "block_start": bs, "block_end": bs + window - 1})
    return variations


def apply_variation(model, tokenizer, K, bs, be):
    if hasattr(rvd, "unpatch_model"):
        try:
            rvd.unpatch_model(model)
        except Exception as e:
            print(f"[warn] rvd.unpatch_model raised: {e}")
    rvd.patch_model(model, bs, be, tokenizer=tokenizer)
    rvd.set_K(model, K)


def passk_unbiased(M, c, n):
    if c <= 0:
        return 0.0
    if M - c < n:
        return 1.0
    return 1.0 - comb(M - c, n) / comb(M, n)


def run_temp_bon(model, processor, items, base_v, n_temp, temperature, batch_size,
                 max_new_tokens, device, estimator="unbiased", verbose=False):
    N = len(items)
    apply_variation(model, processor.tokenizer,
                    base_v["K"], base_v["block_start"], base_v["block_end"])
    correct_counts, solved_first = [], []
    for si, it in enumerate(items):
        preds = run_samples(model, processor, it["image"], it["prompt"],
                            max_new_tokens, device, n_temp, temperature, batch_size)
        g = [verify(p, it["answer"], it["question"]) for p in preds]
        correct_counts.append(sum(g))
        solved_first.append(next((j + 1 for j, x in enumerate(g) if x), float("inf")))
        if verbose and (si + 1) % 50 == 0:
            print(f"    [temp-bon] {si+1}/{N} (mean correct/{n_temp}: "
                  f"{sum(correct_counts)/len(correct_counts):.2f})")
    curve = []
    for n in range(1, n_temp + 1):
        if estimator == "prefix":
            curve.append(sum(1 for f in solved_first if f <= n) / N if N else 0.0)
        else:
            curve.append(sum(passk_unbiased(n_temp, c, n) for c in correct_counts) / N if N else 0.0)
    return curve


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[load] {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa")
    model.eval()
    input_device = next(model.parameters()).device
    print(f"[info] inputs -> {input_device}")

    if args.lora_dir:
        from peft import PeftModel
        print(f"[lora] {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        model.eval()

    print(f"[data] load_from_disk: {args.dataset_dir}")
    ds = load_ds(args.dataset_dir, args.dataset_split)
    print(f"[data] columns: {ds.column_names}  size: {len(ds)}")

    indices = list(range(len(ds)))[:args.max_samples]
    items = []
    for i in indices:
        ex = ds[i]
        q = str(_get(ex, ["question"], "")).strip()
        items.append({
            "idx": i,
            "image": get_image(ex),
            "question": q,
            "answer": _get(ex, ["answer"], None),
            "prompt": build_prompt(q, args.cot),
        })
    bad = sum(1 for it in items if it["answer"] is None)
    if bad:
        print(f"[warn] {bad} samples have no gold answer — score 0.")
    N = len(items)
    print(f"[eval] N={N}  cot={args.cot}")

    variations = build_variations(args.K_eval, args.block_min, args.block_max, args.block_window)
    n_var = len(variations)
    print(f"[cascade] {n_var} variations")

    remaining = list(range(N))
    solved_by = {}
    per_variation_log = []

    for v_idx, v in enumerate(variations):
        if not remaining:
            print(f"[cascade] all {N} solved — stopping at variation {v_idx}")
            break
        K, bs, be = v["K"], v["block_start"], v["block_end"]
        print(f"\n[var {v_idx:02d}] K={K} blocks=[{bs},{be}]  trying {len(remaining)}")
        apply_variation(model, processor.tokenizer, K, bs, be)

        newly, still = 0, []
        for s_idx in remaining:
            it = items[s_idx]
            pred = run_one(model, processor, it["image"], it["prompt"],
                           args.max_new_tokens, input_device)
            ok = verify(pred, it["answer"], it["question"])
            if ok:
                newly += 1
                solved_by[s_idx] = {"var_idx": v_idx, "K": K}
            else:
                still.append(s_idx)
            if args.verbose:
                print(f"    s={s_idx} gold={it['answer']!r} pred={extract_answer(pred)!r} "
                      f"{'OK' if ok else 'x'}")
        per_variation_log.append({
            "var_idx": v_idx, "K": K, "block_start": bs, "block_end": be,
            "attempted": len(remaining), "solved": newly,
            "cumulative_solved": len(solved_by),
        })
        print(f"[var {v_idx:02d}] solved {newly}/{len(remaining)}  "
              f"cum {len(solved_by)}/{N} = {len(solved_by)/N*100:.1f}%")
        remaining = still

    total_solved = len(solved_by)
    final_acc = total_solved / N if N else 0.0

    cascade_curve, last = [], 0.0
    for n in range(n_var):
        if n < len(per_variation_log):
            last = per_variation_log[n]["cumulative_solved"] / N if N else 0.0
        cascade_curve.append(last)
    x_cascade = list(range(1, n_var + 1))
    base_log = next((r for r in per_variation_log if r["K"] == 1), None)
    base_acc = (base_log["solved"] / N) if (base_log and N) else 0.0

    temp_curve, n_temp = None, None
    if not args.skip_temp_bon:
        n_temp = args.temp_bon_samples if args.temp_bon_samples is not None else n_var
        base_v = next((v for v in variations if v["K"] == 1), variations[0])
        print(f"\n[temp-bon] temp={args.temp} ({args.temp_bon_estimator}): "
              f"~{N*n_temp} generations x {args.max_new_tokens} tok")
        temp_curve = run_temp_bon(model, processor, items, base_v, n_temp, args.temp,
                                  args.temp_bon_batch, args.max_new_tokens, input_device,
                                  args.temp_bon_estimator, args.verbose)

    print("\n=== RealWorldQA verifiable best-of-n summary ===")
    print(f"N={N}  final solved={total_solved}/{N}={final_acc*100:.2f}%  unsolved={len(remaining)}")
    print(f"K=1 base (greedy) acc = {base_acc*100:.2f}%")
    if temp_curve is not None:
        print(f"temp={args.temp} best-of-1={temp_curve[0]*100:.2f}%  "
              f"best-of-{n_temp}={temp_curve[-1]*100:.2f}%")
    for row in per_variation_log:
        print(f"  var{row['var_idx']:>2} K={row['K']:>2} [{row['block_start']},{row['block_end']}] "
              f"solved {row['solved']:>4}  cum {row['cumulative_solved']/N*100:5.1f}%")

    # ---- Plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(x_cascade, [v*100 for v in cascade_curve], marker="o", markersize=4, linewidth=1.8,
             label="Cascade best-of-n (K / block search)")
    if temp_curve is not None:
        plt.plot(list(range(1, len(temp_curve)+1)), [v*100 for v in temp_curve],
                 marker="s", markersize=4, linewidth=1.8,
                 label=f"Base temp={args.temp} best-of-n ({args.temp_bon_estimator})")
    plt.axhline(base_acc*100, linestyle="--", color="gray", linewidth=1.5,
                label=f"K=1 base (greedy) = {base_acc*100:.1f}%")
    plt.xlabel("n  (number of variations / samples tried)")
    plt.ylabel("Accuracy (%)")
    plt.title("Best-of-n accuracy vs n (RealWorldQA)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.plot_output, dpi=150)
    plt.close()
    print(f"[plot] wrote {args.plot_output}")

    # ---- TXT
    with Path(args.output).open("w") as f:
        f.write("RealWorldQA Verifiable Best-of-n Evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model:        {args.model}\n")
        if args.lora_dir:
            f.write(f"LoRA:         {args.lora_dir}\n")
        f.write(f"Dataset dir:  {args.dataset_dir}  split={args.dataset_split}\n")
        f.write(f"N samples:    {N}\n")
        f.write(f"cot:          {args.cot}\n")
        f.write(f"K_eval:       {args.K_eval}\n")
        f.write(f"block range:  [{args.block_min}, {args.block_max}] window={args.block_window}\n")
        f.write(f"max_new_tok:  {args.max_new_tokens}\n")
        f.write(f"variations:   {n_var}\n")
        f.write(f"temp base:    {'disabled' if args.skip_temp_bon else f'temp={args.temp}, n_temp={n_temp}, est={args.temp_bon_estimator}'}\n\n")
        f.write(f"FINAL solved: {total_solved}/{N} = {final_acc*100:.2f}%\n")
        f.write(f"K=1 base (greedy): {base_acc*100:.2f}%\n\n")

        f.write("Per-variation\n" + "-"*60 + "\n")
        f.write(f"{'var':>4} | {'K':>3} | {'blocks':>9} | {'attempted':>9} | {'solved':>6} | {'cum%':>7}\n")
        for row in per_variation_log:
            f.write(f"{row['var_idx']:>4} | {row['K']:>3} | "
                    f"[{row['block_start']:>2},{row['block_end']:>2}] | "
                    f"{row['attempted']:>9} | {row['solved']:>6} | "
                    f"{row['cumulative_solved']/N*100:>6.2f}%\n")

        f.write("\nBest-of-n curves (accuracy %)\n" + "-"*60 + "\n")
        f.write(f"K=1 base (greedy): {base_acc*100:.2f}%\n\n")
        if temp_curve is not None:
            f.write(f"{'n':>4} | {'cascade%':>10} | {'temp%':>8}\n")
            for i in range(n_var):
                tval = temp_curve[i]*100 if i < len(temp_curve) else float('nan')
                ts = f"{tval:8.2f}" if tval == tval else f"{'n/a':>8}"
                f.write(f"{i+1:>4} | {cascade_curve[i]*100:10.2f} | {ts}\n")
        else:
            f.write(f"{'n':>4} | {'cascade%':>10}\n")
            for i in range(n_var):
                f.write(f"{i+1:>4} | {cascade_curve[i]*100:10.2f}\n")
        f.write(f"\nPlot saved to: {args.plot_output}\n")
        f.write("\n" + "="*60 + f"\nFINAL: {total_solved}/{N} = {final_acc*100:.2f}%\n")

    print(f"[log] wrote {args.output}")


if __name__ == "__main__":
    main()