"""
Best-of-n (K / block-window) eval on an OPEN-ENDED NUMERIC visual dataset
(default: CountBenchQA). Model-agnostic: the model is chosen by which RVD
backend you import below (Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL).

Difference from the MCQ version (mcq_bon.py)
--------------------------------------------
MCQ datasets are right (1) or wrong (0), so a sample is dropped the first time
any variation gets it right.

Here answers are NUMBERS, so each (sample, variation) gets a CONTINUOUS
closeness score in [0, 1]:

    score = clamp(1 - |pred - gt| / |gt|, 0, 1)        # 'relerr' (default)
    e.g. gt=10, pred=8  -> 1 - 2/10 = 0.80  (80%)
         gt=10, pred=12 -> 1 - 2/10 = 0.80
         gt=10, pred=20 -> 0.00

Best-of-n keeps the HIGHEST score a sample achieved over the n attempts:

    per-sample best-of-n = max(score over the first n variations)
    dataset best-of-n     = mean over samples of that per-sample max

So the cascade tracks a running per-sample MAX. To save compute, a sample is
dropped only once it hits the solve threshold (default 1.0 = exact match);
otherwise it keeps being re-tried by later variations so its max can improve.

The model is asked to put its final integer in \boxed{...}; extraction prefers
the boxed value and falls back to the last number / number-word in the text.

Outputs:
  - Per-variation log (attempted / newly solved / cumulative mean best score)
  - Final mean closeness (best-of-all-variations)
  - A best-of-n plot (PNG):
      x = n (number of variations / samples tried)
      y = mean closeness (%)
        * Cascade best-of-n curve
        * Base model temp>0 best-of-n curve (unbiased expected-max-of-n-subset)
        * Dashed line: K=1 base (greedy) mean closeness
  - A txt log with all of the above + the curve data.

Usage:
    python count_bon.py \
        --model /models/Qwen2.5-VL-7B-Instruct\
        --dataset_dir /data/countqa \
        --max_samples 50 \
        --K_eval 1 2 4 \
        --block_min 0 --block_max 4 --block_window 3 \
        --output countqa_bon_results.txt \
        --plot_output countqa_bon_curve.png --score_metric relerr
"""

import argparse
import io
import re
from collections import defaultdict
from math import comb
from pathlib import Path

import torch
from datasets import load_from_disk, Dataset, DatasetDict
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import matplotlib
matplotlib.use("Agg")  # headless
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
    p.add_argument("--dataset_dir",
                   default="/data/CountBenchQA",
                   required=True, type=str)
    p.add_argument("--dataset_split", default=None)
    p.add_argument("--max_samples", type=int, default=491)

    # cascade parameters
    p.add_argument("--K_eval", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="K values to try, in cascade order (smallest first recommended).")
    p.add_argument("--block_min", type=int, default=0)
    p.add_argument("--block_max", type=int, default=21)
    p.add_argument("--block_window", type=int, default=3,
                   help="Fixed window length (inclusive). Stride is 1.")

    # numeric scoring
    p.add_argument("--score_metric", choices=["relerr", "ratio", "abs"], default="relerr",
                   help="'relerr': 1 - |pred-gt|/|gt| (clamped); gt=10,pred=8 -> 0.80. "
                        "'ratio': min(pred,gt)/max(pred,gt).")
    p.add_argument("--solve_threshold", type=float, default=1.0,
                   help="A sample is 'solved' and dropped from the cascade once its best "
                        "score reaches this (1.0 = exact match). Lower it to drop "
                        "near-perfect samples early and save compute.")

    p.add_argument("--max_new_tokens", type=int, default=120,
                   help="Room for a short rationale + \\boxed{N}.")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--stratified", action="store_true",
                   help="Stratify the subsample across ground-truth counts.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=str, default="count_bon_results.txt")

    # ---- best-of-n plot / temperature baseline
    p.add_argument("--plot_output", type=str, default="count_bon_curve.png")
    p.add_argument("--skip_temp_bon", action="store_true",
                   help="Skip the temperature>0 best-of-n baseline (expensive: "
                        "N * n_temp generations).")
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--temp_bon_samples", type=int, default=None,
                   help="Samples/item for the temp baseline. Default = number of variations.")
    p.add_argument("--temp_bon_batch", type=int, default=8)
    p.add_argument("--temp_bon_estimator", choices=["unbiased", "prefix"], default="prefix",
                   help="'unbiased': expected MAX over all n-subsets of the drawn samples "
                        "(closed form, low variance; generalizes pass@k to scores). "
                        "'prefix': running max of the first n draws (single noisy draw).")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_numeric_ds(dataset_dir: str, split: str | None):
    ds = load_from_disk(dataset_dir)
    if isinstance(ds, DatasetDict) or (not isinstance(ds, Dataset) and hasattr(ds, "keys")):
        keys = list(ds.keys())
        if split is None:
            split = "test" if "test" in keys else ("train" if "train" in keys else keys[0])
        print(f"[data] DatasetDict found, using split='{split}' (available: {keys})")
        ds = ds[split]
    return ds


# Flexible field names so CountBenchQA ('number') and similar datasets
# ('answer'/'count'/'gt_answer'/'label') both work.
_FIELDS = {
    "image":    ["image", "Image", "img"],
    "question": ["question", "Question", "query"],
    "answer":   ["number", "answer","answers", "count", "gt_answer", "label", "Answer"],
    "idx":      ["idx", "Index", "index", "id"],
}


def _get(ex, kind, default=None):
    for k in _FIELDS[kind]:
        if k in ex and ex[k] is not None:
            return ex[k]
    return default


def get_image(ex):
    img = _get(ex, "image")
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
# Prompt, number extraction & scoring
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "{question}\n"
    "Answer with a single integer (the count). "
    "Put your final answer inside \\boxed{{}}."
)

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

_BOXED_RE = re.compile(r"\\boxed\s*\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _to_float(tok: str):
    tok = tok.strip().replace(",", "")
    try:
        return float(tok)
    except ValueError:
        return _NUM_WORDS.get(tok.lower())  # may be None


def extract_number(text: str):
    """Prefer the value inside \\boxed{...}; fall back to the last number, then
    the last number-word. Returns float or None."""
    if not text:
        return None
    m = _BOXED_RE.search(text)
    if m:
        inner = m.group(1)
        nm = _NUM_RE.search(inner)
        if nm:
            return float(nm.group().replace(",", ""))
        w = _to_float(inner)
        if w is not None:
            return float(w)
    nums = _NUM_RE.findall(text)
    if nums:
        return float(nums[-1].replace(",", ""))
    for tok in reversed(re.findall(r"[a-zA-Z]+", text.lower())):
        if tok in _NUM_WORDS:
            return float(_NUM_WORDS[tok])
    return None


def gold_number(answer):
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        return float(answer)
    return extract_number(str(answer))


def closeness_score(pred, gt, metric="relerr") -> float:
    """Continuous correctness in [0, 1]."""
    if pred is None or gt is None:
        return 0.0
    pred, gt = float(pred), float(gt)
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    if metric == "ratio":
        lo, hi = min(pred, gt), max(pred, gt)
        return max(0.0, lo / hi) if hi > 0 else 0.0
    if metric=="abs":
        if gt==pred:
            return 1.0
        else:
            return 0
    # relerr (default)
    return max(0.0, 1.0 - abs(pred - gt) / abs(gt))


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def _build_inputs(processor, image, question, device):
    user_text = PROMPT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": user_text},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        # Qwen3.5-VL only: add `enable_thinking=False,` here to disable thinking mode.
    )
    inputs = processor(text=[text], images=[image],
                       max_pixels=512 * 28 * 28, return_tensors="pt").to(device)
    return inputs


@torch.no_grad()
def run_one(model, processor, image, question, max_new_tokens, device):
    """Greedy single decode (used by the cascade)."""
    inputs = _build_inputs(processor, image, question, device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature= 0.6)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


@torch.no_grad()
def run_samples(model, processor, image, question, max_new_tokens, device,
                n_samples, temperature, batch_size):
    """Draw `n_samples` stochastic decodes at `temperature` for one item."""
    inputs = _build_inputs(processor, image, question, device)
    preds = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, num_return_sequences=b,
        )
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
        raise ValueError(f"block range [{block_min},{block_max}] too small for window={window}")
    variations = []
    for K in K_eval:
        if K == 1:
            variations.append({"K": int(K), "block_start": 1, "block_end": 3})
            continue
        for bs in range(block_min, block_max - window + 2):
            be = bs + window - 1
            variations.append({"K": int(K), "block_start": bs, "block_end": be})
    return variations


def apply_variation(model, tokenizer, K, block_start, block_end):
    if hasattr(rvd, "unpatch_model"):
        try:
            rvd.unpatch_model(model)
        except Exception as e:
            print(f"[warn] rvd.unpatch_model raised: {e}")
    rvd.patch_model(model, block_start, block_end, tokenizer=tokenizer)
    rvd.set_K(model, K)


# -----------------------------------------------------------------------------
# Best-of-n score estimators (continuous generalization of pass@k)
# -----------------------------------------------------------------------------

def best_of_n_score_unbiased(sorted_scores, M, n):
    """
    Expected MAX of a uniformly random size-n subset (without replacement) drawn
    from M scores. With sorted s_1<=...<=s_M:
        E[max] = sum_k  s_k * C(k-1, n-1) / C(M, n)
    Uses all M draws for every n -> low variance. For binary scores this reduces
    to the pass@k estimator.
    """
    if n > M:
        n = M
    denom = comb(M, n)
    total = 0.0
    for k in range(n, M + 1):          # C(k-1, n-1) == 0 for k < n
        total += sorted_scores[k - 1] * comb(k - 1, n - 1)
    return total / denom


def run_temp_bon(model, processor, items, base_v, n_temp, temperature, batch_size,
                 max_new_tokens, device, metric="relerr", estimator="unbiased",
                 verbose=False):
    """
    For each item, draw n_temp samples at `temperature` from the base config and
    score each. Build the best-of-n closeness curve:
      'unbiased': expected max over all n-subsets (recommended).
      'prefix':   running max of the first n draws (single noisy realization).
    Returns (temp_bon_curve, per_item_scores).
    """
    N = len(items)
    apply_variation(model, processor.tokenizer,
                    base_v["K"], base_v["block_start"], base_v["block_end"])

    per_item_scores = []  # list of per-item score lists (len n_temp)
    for si, it in enumerate(items):
        preds = run_samples(
            model, processor, it["image"], it["question"],
            max_new_tokens, device, n_temp, temperature, batch_size,
        )
        scores = [closeness_score(extract_number(p), it["gold"], metric) for p in preds]
        per_item_scores.append(scores)
        if verbose and (si + 1) % 50 == 0:
            mean_best = sum(max(s) for s in per_item_scores) / len(per_item_scores)
            print(f"    [temp-bon] {si + 1}/{N} items (mean best-of-{n_temp}: {mean_best:.3f})")

    temp_bon_curve = []
    if estimator == "unbiased":
        sorted_all = [sorted(s) for s in per_item_scores]
        for n in range(1, n_temp + 1):
            acc = sum(best_of_n_score_unbiased(ss, n_temp, n) for ss in sorted_all) / N if N else 0.0
            temp_bon_curve.append(acc)
    else:  # prefix
        for n in range(1, n_temp + 1):
            acc = sum(max(s[:n]) for s in per_item_scores) / N if N else 0.0
            temp_bon_curve.append(acc)

    return temp_bon_curve, per_item_scores


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # ---- Model
    print(f"[load] {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    input_device = next(model.parameters()).device
    print(f"[info] sending inputs to {input_device}")

    if args.lora_dir:
        from peft import PeftModel
        print(f"[lora] loading {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        model.eval()

    # ---- Data
    print(f"[data] load_from_disk: {args.dataset_dir}")
    ds = load_numeric_ds(args.dataset_dir, args.dataset_split)
    print(f"[data] columns: {ds.column_names}")
    print(f"[data] full size: {len(ds)}")

    indices = list(range(len(ds)))

    if args.stratified and args.max_samples < len(indices):
        buckets = defaultdict(list)
        for i in indices:
            g = gold_number(_get(ds[i], "answer"))
            buckets[str(g)].append(i)
        per_bucket = max(1, args.max_samples // max(1, len(buckets)))
        picked = []
        for _, idxs in buckets.items():
            picked.extend(idxs[:per_bucket])
        indices = picked[:args.max_samples]
        print(f"[stratified] picked {len(indices)} across {len(buckets)} count-buckets "
              f"(~{per_bucket}/bucket)")
    else:
        indices = indices[:args.max_samples]

    items = []
    for i in indices:
        ex = ds[i]
        items.append({
            "idx":      _get(ex, "idx", i),
            "image":    get_image(ex),
            "question": str(_get(ex, "question", "")).strip(),
            "gold":     gold_number(_get(ex, "answer")),
        })

    bad = sum(1 for it in items if it["gold"] is None)
    if bad:
        print(f"[warn] {bad} samples have unresolvable gold numbers — they score 0.")

    N = len(items)
    print(f"[eval] N={N}  metric={args.score_metric}  solve_threshold={args.solve_threshold}")

    # ---- Cascade plan
    variations = build_variations(args.K_eval, args.block_min, args.block_max, args.block_window)
    n_var = len(variations)
    print(f"[cascade] {n_var} variations:")
    for v in variations:
        print(f"   K={v['K']:>2}  blocks=[{v['block_start']},{v['block_end']}]")

    # ---- Cascade run (track per-sample running MAX score)
    best_score = [0.0] * N          # best closeness any tried variation achieved
    best_var = [None] * N           # which variation gave that best
    remaining = list(range(N))      # samples not yet at solve_threshold
    per_variation_log = []
    # var_idx -> sum of (improvement in best score) it contributed, for diagnostics
    var_contrib = defaultdict(float)

    for v_idx, v in enumerate(variations):
        if not remaining:
            print(f"[cascade] all {N} samples reached threshold — stopping at variation {v_idx}")
            break

        K, bs, be = v["K"], v["block_start"], v["block_end"]
        print(f"\n[var {v_idx:02d}] K={K} blocks=[{bs},{be}]  "
              f"attempting {len(remaining)} sample(s) below threshold")

        apply_variation(model, processor.tokenizer, K, bs, be)

        newly_solved = 0
        still_open = []
        for sample_idx in remaining:
            it = items[sample_idx]
            pred = run_one(model, processor, it["image"], it["question"],
                           args.max_new_tokens, input_device)
            s = closeness_score(extract_number(pred), it["gold"], args.score_metric)

            if s > best_score[sample_idx]:
                var_contrib[v_idx] += s - best_score[sample_idx]
                best_score[sample_idx] = s
                best_var[sample_idx] = v_idx

            if best_score[sample_idx] >= args.solve_threshold:
                newly_solved += 1
            else:
                still_open.append(sample_idx)

            if args.verbose:
                print(f"    sample={sample_idx} gold={it['gold']} "
                      f"pred={extract_number(pred)} score={s:.2f} "
                      f"best={best_score[sample_idx]:.2f}  raw={pred.strip()!r:.60}")

        cum_mean = sum(best_score) / N if N else 0.0
        per_variation_log.append({
            "var_idx": v_idx, "K": K, "block_start": bs, "block_end": be,
            "attempted": len(remaining), "newly_solved": newly_solved,
            "cum_mean_best": cum_mean,
        })
        print(f"[var {v_idx:02d}] newly solved (>= {args.solve_threshold}): {newly_solved}  "
              f"cumulative mean best closeness: {cum_mean*100:.2f}%")
        remaining = still_open

    # ---- Aggregate
    final_mean = sum(best_score) / N if N else 0.0
    n_solved = sum(1 for s in best_score if s >= args.solve_threshold)
    exact = sum(1 for it, s in zip(items, best_score) if s >= 1.0)

    # ---- Best-of-n curves
    cascade_curve = []
    last = 0.0
    for n in range(n_var):
        if n < len(per_variation_log):
            last = per_variation_log[n]["cum_mean_best"]
        cascade_curve.append(last)
    x_cascade = list(range(1, n_var + 1))

    base_log = next((r for r in per_variation_log if r["K"] == 1), None)
    # K=1 greedy mean closeness == mean best after the (first) K=1 variation IF it
    # is variation 0; in the general case recompute from its cumulative entry.
    base_greedy_mean = base_log["cum_mean_best"] if base_log else 0.0

    temp_bon_curve = None
    n_temp = None
    if not args.skip_temp_bon:
        n_temp = args.temp_bon_samples if args.temp_bon_samples is not None else n_var
        base_v = next((v for v in variations if v["K"] == 1), variations[0])
        print(f"\n[temp-bon] temp={args.temp} best-of-n baseline "
              f"({args.temp_bon_estimator}): N={N} x n_temp={n_temp} "
              f"= ~{N*n_temp} generations "
              f"(base K={base_v['K']} blocks=[{base_v['block_start']},{base_v['block_end']}])")
        temp_bon_curve, _ = run_temp_bon(
            model, processor, items, base_v, n_temp, args.temp, args.temp_bon_batch,
            args.max_new_tokens, input_device,
            metric=args.score_metric, estimator=args.temp_bon_estimator, verbose=args.verbose,
        )

    # ---- Print summary
    print("\n=== Numeric cascade summary ===")
    print(f"N = {N}   metric = {args.score_metric}")
    print(f"Final mean closeness (best-of-all) = {final_mean*100:.2f}%")
    print(f"Solved (>= {args.solve_threshold}) = {n_solved}/{N}   exact = {exact}/{N}")
    print(f"K=1 base (greedy) mean closeness   = {base_greedy_mean*100:.2f}%")
    if temp_bon_curve is not None:
        print(f"temp={args.temp} best-of-1  = {temp_bon_curve[0]*100:.2f}%")
        print(f"temp={args.temp} best-of-{n_temp} = {temp_bon_curve[-1]*100:.2f}%")

    header = f"{'var':>4} | {'K':>3} | {'blocks':>9} | {'attempted':>9} | {'solved':>6} | {'cum_mean%':>9}"
    print(header)
    print("-" * len(header))
    for row in per_variation_log:
        print(f"{row['var_idx']:>4} | {row['K']:>3} | "
              f"[{row['block_start']:>2},{row['block_end']:>2}] | "
              f"{row['attempted']:>9} | {row['newly_solved']:>6} | "
              f"{row['cum_mean_best']*100:>8.2f}%")

    # ---- Plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(x_cascade, [v * 100 for v in cascade_curve],
             marker="o", markersize=4, linewidth=1.8,
             label="Cascade best-of-n (K / block-window search)")
    if temp_bon_curve is not None:
        x_temp = list(range(1, len(temp_bon_curve) + 1))
        plt.plot(x_temp, [v * 100 for v in temp_bon_curve],
                 marker="s", markersize=4, linewidth=1.8,
                 label=f"Base model temp={args.temp} best-of-n ({args.temp_bon_estimator})")
    plt.axhline(base_greedy_mean * 100, linestyle="--", color="gray", linewidth=1.5,
                label=f"K=1 base (greedy) = {base_greedy_mean*100:.1f}%")
    plt.xlabel("n  (number of variations / samples tried)")
    plt.ylabel(f"Mean closeness (%)  [{args.score_metric}]")
    plt.title("Best-of-n closeness vs n (open-ended numeric)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plot_path = Path(args.plot_output)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[plot] wrote {plot_path}")

    # ---- TXT log
    out_path = Path(args.output)
    with out_path.open("w") as f:
        f.write("Open-ended Numeric Cascade Evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model:          {args.model}\n")
        if args.lora_dir:
            f.write(f"LoRA:           {args.lora_dir}\n")
        f.write(f"Dataset dir:    {args.dataset_dir}\n")
        f.write(f"N samples:      {N}\n")
        f.write(f"Score metric:   {args.score_metric}\n")
        f.write(f"Solve thresh:   {args.solve_threshold}\n")
        f.write(f"K_eval:         {args.K_eval}\n")
        f.write(f"block range:    [{args.block_min}, {args.block_max}]  window={args.block_window}\n")
        f.write(f"variations:     {n_var}\n")
        f.write(f"temp baseline:  {'disabled' if args.skip_temp_bon else f'temp={args.temp}, n_temp={n_temp}, estimator={args.temp_bon_estimator}'}\n\n")

        f.write(f"FINAL mean closeness (best-of-all): {final_mean*100:.2f}%\n")
        f.write(f"Solved (>= {args.solve_threshold}): {n_solved}/{N}   exact: {exact}/{N}\n")
        f.write(f"K=1 base (greedy) mean closeness:   {base_greedy_mean*100:.2f}%\n\n")

        f.write("Per-variation results\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'var':>4} | {'K':>3} | {'blocks':>9} | {'attempted':>9} | "
                f"{'solved':>6} | {'cum_mean%':>10}\n")
        for row in per_variation_log:
            f.write(f"{row['var_idx']:>4} | {row['K']:>3} | "
                    f"[{row['block_start']:>2},{row['block_end']:>2}] | "
                    f"{row['attempted']:>9} | {row['newly_solved']:>6} | "
                    f"{row['cum_mean_best']*100:>9.2f}%\n")
        unrun = n_var - len(per_variation_log)
        if unrun > 0:
            f.write(f"\n({unrun} variation(s) not executed — cascade exited early)\n")

        f.write("\nBest-of-n curves (mean closeness %)\n")
        f.write("-" * 60 + "\n")
        f.write(f"K=1 base (greedy): {base_greedy_mean*100:.2f}%\n\n")
        if temp_bon_curve is not None:
            f.write(f"{'n':>4} | {'cascade_bon%':>13} | {'temp_bon%':>11}\n")
            for i in range(n_var):
                casc = cascade_curve[i] * 100
                tval = temp_bon_curve[i] * 100 if i < len(temp_bon_curve) else float('nan')
                tstr = f"{tval:11.2f}" if tval == tval else f"{'n/a':>11}"
                f.write(f"{i+1:>4} | {casc:13.2f} | {tstr}\n")
            if len(temp_bon_curve) > n_var:
                for i in range(n_var, len(temp_bon_curve)):
                    f.write(f"{i+1:>4} | {'n/a':>13} | {temp_bon_curve[i]*100:11.2f}\n")
        else:
            f.write(f"{'n':>4} | {'cascade_bon%':>13}\n")
            for i in range(n_var):
                f.write(f"{i+1:>4} | {cascade_curve[i]*100:13.2f}\n")

        f.write(f"\nPlot saved to: {plot_path}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"FINAL: mean closeness {final_mean*100:.2f}%  "
                f"(solved {n_solved}/{N}, exact {exact}/{N})\n")

    print(f"[log] wrote {out_path}")


if __name__ == "__main__":
    main()