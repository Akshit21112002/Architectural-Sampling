"""
Best-of-n (K / block-window) eval on a MULTIPLE-CHOICE (MCQ) visual dataset.

Works for any MCQ dataset that exposes an image, a question, a list of choices
and a letter/text answer — e.g. CV-Bench, AI2D, MMMU — just point --dataset_dir
at the load_from_disk folder. Model-agnostic: the model is chosen by which RVD
backend you import below (Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL).

Per-sample best-of-n search:
  - Try variation 1: K=K_eval[0], blocks=[bmin, bmin+win-1]
  - If wrong, try variation 2: same K, blocks=[bmin+1, bmin+win]  (stride 1)
  - ... slide window up to [bmax-win+1, bmax]
  - Then bump K to K_eval[1] and restart the window scan
  - First variation that answers correctly "solves" the sample; it is dropped.

To avoid reloading the model, the OUTER loop iterates variations and the INNER
loop iterates only over samples that are still unsolved. Each sample is
answered correctly by AT MOST one variation (the first that gets it right).

Outputs:
  - Per-variation counts (attempted / newly solved / cumulative)
  - Final total accuracy (sum of correctly-answered samples / N)
  - Per-task breakdown
  - A txt log of all of the above
  - A "best-of-n" plot (PNG):
      x-axis = n (number of variations / samples tried)
      y-axis = accuracy (%)
        * Cascade best-of-n curve (cumulative_solved / N after n variations)
        * Base model temp>0 best-of-n curve (same n, sampled n times)
        * Dashed horizontal line: K=1 base (greedy) accuracy
  - Curve data appended to the txt log.

Usage:
    python mcq_bon.py \
        --model /models/Qwen3-VL-2B-Instruct \
        --dataset_dir /data/CV-Bench \
        --max_samples 2638 \
        --K_eval 1 2 4 \
        --block_min 0 --block_max 5 --block_window 3 --temp 0.6 \
        --output mcq_bon_results.txt \
        --plot_output mcq_bon_curve.png
"""

import argparse
import io
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk, Dataset, DatasetDict
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import matplotlib
matplotlib.use("Agg")  # headless / no display
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
                   default="/data/CV-Bench",
                   required=True, type=str,
                   help="load_from_disk folder of an MCQ dataset "
                        "(e.g. CV-Bench, AI2D, MMMU).")
    p.add_argument("--dataset_split", default=None)
    p.add_argument("--max_samples", type=int, default=200)

    # cascade parameters
    p.add_argument("--K_eval", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="K values to try, in cascade order (smallest first recommended).")
    p.add_argument("--block_min", type=int, default=0,
                   help="Lowest block index considered (inclusive).")
    p.add_argument("--block_max", type=int, default=21,
                   help="Highest block index considered (inclusive).")
    p.add_argument("--block_window", type=int, default=3,
                   help="Fixed window length (inclusive). Stride is 1.")

    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--task_filter", nargs="+", default=None)
    p.add_argument("--stratified", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=str, default="mcq_bon_results.txt",
                   help="Where to write the best-of-n summary log.")

    # ---- best-of-n plot / temperature baseline
    p.add_argument("--plot_output", type=str, default="mcq_bon_curve.png",
                   help="Where to write the best-of-n comparison plot (PNG).")
    p.add_argument("--skip_temp_bon", action="store_true",
                   help="Skip the temperature>0 best-of-n baseline (it is expensive: "
                        "N * n_temp generations).")
    p.add_argument("--temp", type=float, default=0.6,
                   help="Sampling temperature for the best-of-n baseline.")
    p.add_argument("--temp_bon_samples", type=int, default=None,
                   help="Number of stochastic samples per item for the temp baseline. "
                        "Defaults to the number of cascade variations (so both curves "
                        "share the same x-axis).")
    p.add_argument("--temp_bon_batch", type=int, default=8,
                   help="num_return_sequences per generate() call for the temp baseline.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Data (unchanged from your script)
# -----------------------------------------------------------------------------

def load_cvbench(dataset_dir: str, split: str | None):
    ds = load_from_disk(dataset_dir)
    if isinstance(ds, DatasetDict) or (not isinstance(ds, Dataset) and hasattr(ds, "keys")):
        keys = list(ds.keys())
        if split is None:
            split = "test" if "test" in keys else ("train" if "train" in keys else keys[0])
        print(f"[data] DatasetDict found, using split='{split}' (available: {keys})")
        ds = ds[split]
    return ds


_FIELDS = {
    "image":    ["image", "Image", "img"],
    "question": ["question", "Question"],
    "choices":  ["choices", "options", "Choices", "Options"],
    "answer":   ["answer", "Answer", "correct_answer"],
    "task":     ["task", "Task"],
    "type":     ["type", "Type"],
    "source":   ["source", "Source"],
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
# Prompt & grading (unchanged)
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "{question}\n"
    "{options}\n"
    "Answer with the option's letter from the given choices directly."
)

_LETTERS = "ABCDEFGH"



def format_options(choices):
    if not isinstance(choices, (list, tuple)):
        return str(choices)
    return "\n".join(f"({_LETTERS[i]}) {c}" for i, c in enumerate(choices))


_LETTER_RE = re.compile(r"\(?\b([A-Ha-h])\b\)?")


def extract_letter(s: str) -> str | None:
    if not s:
        return None
    m = _LETTER_RE.search(s)
    return m.group(1).upper() if m else None


def gold_letter(answer: str, choices) -> str | None:
    if answer is None:
        return None
    answer = str(answer).strip()
    if len(answer) == 1 and answer.upper() in _LETTERS:
        return answer.upper()
    m = re.match(r"^\(?([A-Ha-h])\)?$", answer)
    if m:
        return m.group(1).upper()
    if isinstance(choices, (list, tuple)):
        for i, c in enumerate(choices):
            if str(c).strip().lower() == answer.lower():
                return _LETTERS[i]
        for i, c in enumerate(choices):
            if answer.lower() in str(c).strip().lower():
                return _LETTERS[i]
    return None


def grade(pred_text: str, gold_l: str | None) -> int:
    if gold_l is None:
        return 0
    pl = extract_letter(pred_text)
    return int(pl is not None and pl == gold_l)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def _build_inputs(processor, image, question, options, device):
    user_text = PROMPT_TEMPLATE.format(question=question, options=options)
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
def run_one(model, processor, image, question, options, max_new_tokens, device):
    """Greedy single decode (used by the cascade)."""
    inputs = _build_inputs(processor, image, question, options, device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,temperature=0.6)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


@torch.no_grad()
def run_samples(model, processor, image, question, options, max_new_tokens, device,
                n_samples, temperature, batch_size):
    """
    Draw `n_samples` stochastic decodes for a single item at the given temperature.
    Generates in chunks of `batch_size` (via num_return_sequences) to bound memory.
    Returns a list of `n_samples` decoded strings.
    """
    inputs = _build_inputs(processor, image, question, options, device)
    preds = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=b,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        preds.extend(processor.tokenizer.batch_decode(gen, skip_special_tokens=True))
        remaining -= b
    return preds


# -----------------------------------------------------------------------------
# Cascade helpers
# -----------------------------------------------------------------------------

def build_variations(K_eval, block_min, block_max, window):
    """
    Generate the cascade order. For each K (in given order), slide a window of
    fixed length `window` from block_min..block_max with stride 1.
    Returns list of dicts: {K, block_start, block_end}.
    """
    if window <= 0:
        raise ValueError("--block_window must be >= 1")
    if block_max < block_min + window - 1:
        raise ValueError(
            f"block range [{block_min},{block_max}] too small for window={window}"
        )
    variations = []
    for K in K_eval:
        if K==1:
            variations.append({"K": int(K), "block_start": 1, "block_end": 3})
            continue
        for bs in range(block_min, block_max - window + 2):
            be = bs + window - 1
            variations.append({"K": int(K), "block_start": bs, "block_end": be})
    return variations


def apply_variation(model, tokenizer, K, block_start, block_end):
    """
    (Re)patch the model for a new (K, block_start, block_end) without reloading
    weights. Tries rvd.unpatch_model first if it exists; otherwise just calls
    rvd.patch_model again and assumes it overwrites the previous patch.

    NOTE: if your `rvd` does NOT support repatching cleanly, add an
    `unpatch_model(model)` to it that restores the original forward hooks /
    forward methods, and this will Just Work.
    """
    if hasattr(rvd, "unpatch_model"):
        try:
            rvd.unpatch_model(model)
        except Exception as e:
            print(f"[warn] rvd.unpatch_model raised: {e}")
    rvd.patch_model(model, block_start, block_end, tokenizer=tokenizer)
    rvd.set_K(model, K)


# -----------------------------------------------------------------------------
# Best-of-n baseline (temperature sampling on the K=1 base config)
# -----------------------------------------------------------------------------

def run_temp_bon(model, processor, items, base_v, n_temp, temperature, batch_size,
                 max_new_tokens, device, verbose=False):
    """
    For each item, draw `n_temp` samples at `temperature` from the base config,
    and record the index (1-based) of the first sample that answers correctly.

    Returns:
        temp_bon_curve: list of length n_temp; temp_bon_curve[n-1] = best-of-n accuracy
        per_task_first: dict task -> list of first-correct indices (or inf)
    """
    N = len(items)
    apply_variation(model, processor.tokenizer,
                    base_v["K"], base_v["block_start"], base_v["block_end"])

    solved_first = []  # first 1-based sample index that is correct, else inf
    per_task_first = defaultdict(list)
    for si, it in enumerate(items):
        preds = run_samples(
            model, processor,
            it["image"], it["question"], it["options"],
            max_new_tokens, device,
            n_temp, temperature, batch_size,
        )
        first = float("inf")
        for j, p in enumerate(preds):
            if grade(p, it["gold_letter"]):
                first = j + 1
                break
        solved_first.append(first)
        per_task_first[it["task"]].append(first)

        if verbose and (si + 1) % 50 == 0:
            done = sum(1 for f in solved_first if f <= n_temp)
            print(f"    [temp-bon] {si + 1}/{N} items "
                  f"(best-of-{n_temp} solved so far: {done})")

    temp_bon_curve = []
    for n in range(1, n_temp + 1):
        c = sum(1 for f in solved_first if f <= n)
        temp_bon_curve.append(c / N if N else 0.0)
    return temp_bon_curve, per_task_first


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
    ds = load_cvbench(args.dataset_dir, args.dataset_split)
    print(f"[data] columns: {ds.column_names}")
    print(f"[data] full size: {len(ds)}")

    indices = list(range(len(ds)))

    if args.task_filter is not None:
        wanted = set(t.lower() for t in args.task_filter)
        indices = [i for i in indices
                   if str(_get(ds[i], "task", "")).lower() in wanted]
        print(f"[filter] task in {args.task_filter}: {len(indices)} samples")

    if args.stratified and args.max_samples < len(indices):
        buckets = defaultdict(list)
        for i in indices:
            t = str(_get(ds[i], "task", "Unknown"))
            buckets[t].append(i)
        per_task = max(1, args.max_samples // max(1, len(buckets)))
        picked = []
        for t, idxs in buckets.items():
            picked.extend(idxs[:per_task])
        indices = picked[:args.max_samples]
        print(f"[stratified] picked {len(indices)} samples across "
              f"{len(buckets)} tasks (~{per_task}/task)")
    else:
        indices = indices[:args.max_samples]

    items = []
    for i in indices:
        ex = ds[i]
        choices = _get(ex, "choices") or []
        items.append({
            "idx":      _get(ex, "idx", i),
            "task":     str(_get(ex, "task", "Unknown")),
            "type":     str(_get(ex, "type", "?")),
            "image":    get_image(ex),
            "question": str(_get(ex, "question", "")).strip(),
            "options":  format_options(choices),
            "gold_letter": gold_letter(_get(ex, "answer"), choices),
        })

    bad = sum(1 for it in items if it["gold_letter"] is None)
    if bad:
        print(f"[warn] {bad} samples have unresolvable gold answers — they'll count as wrong.")

    N = len(items)
    print(f"[eval] N={N}")

    # ---- Cascade plan
    variations = build_variations(
        args.K_eval, args.block_min, args.block_max, args.block_window
    )
    print(f"[cascade] {len(variations)} variations:")
    for v in variations:
        print(f"   K={v['K']:>2}  blocks=[{v['block_start']},{v['block_end']}]")

    # ---- Cascade run
    # remaining: list of sample indices into `items` not yet solved
    # solved_by[sample_idx] = (variation_idx_in_list, K, bs, be)
    remaining = list(range(N))
    solved_by = {}
    per_variation_log = []  # list of dicts (one per variation actually run)

    for v_idx, v in enumerate(variations):
        if not remaining:
            print(f"[cascade] all {N} samples solved — stopping early at variation {v_idx}")
            break

        K, bs, be = v["K"], v["block_start"], v["block_end"]
        print(f"\n[var {v_idx:02d}] K={K} blocks=[{bs},{be}]  "
              f"attempting {len(remaining)} unsolved sample(s)")

        apply_variation(model, processor.tokenizer, K, bs, be)

        newly_solved = []
        still_wrong = []

        for sample_idx in remaining:
            it = items[sample_idx]
            pred = run_one(
                model, processor,
                it["image"], it["question"], it["options"],
                args.max_new_tokens, input_device,
            )
            c = grade(pred, it["gold_letter"])
            if c:
                newly_solved.append(sample_idx)
                solved_by[sample_idx] = {
                    "var_idx": v_idx, "K": K, "bs": bs, "be": be,
                    "task": it["task"],
                }
            else:
                still_wrong.append(sample_idx)

            if args.verbose:
                pred_letter = extract_letter(pred)
                print(f"    sample={sample_idx} task={it['task']:<10} "
                      f"gold={it['gold_letter']} pred={pred_letter} "
                      f"{'✓' if c else '✗'}  raw={pred.strip()!r:<40}")

        per_variation_log.append({
            "var_idx":    v_idx,
            "K":          K,
            "block_start": bs,
            "block_end":   be,
            "attempted":  len(remaining),
            "solved":     len(newly_solved),
            "cumulative_solved": len(solved_by),
        })
        print(f"[var {v_idx:02d}] solved {len(newly_solved)}/{len(remaining)}  "
              f"(cumulative: {len(solved_by)}/{N} = "
              f"{len(solved_by)/N*100:.1f}%)")

        remaining = still_wrong

    # ---- Aggregate
    total_solved = len(solved_by)
    final_acc = total_solved / N if N else 0.0

    # per-task breakdown
    task_total = defaultdict(int)
    task_solved = defaultdict(int)
    for it in items:
        task_total[it["task"]] += 1
    for sample_idx, info in solved_by.items():
        task_solved[info["task"]] += 1

    # which variation solved how many, per task (useful diagnostic)
    var_task = defaultdict(lambda: defaultdict(int))
    for info in solved_by.values():
        var_task[info["var_idx"]][info["task"]] += 1

    # -------------------------------------------------------------------------
    # Best-of-n curves
    # -------------------------------------------------------------------------
    # Cascade best-of-n curve: cumulative_solved / N after n variations.
    # x runs over ALL planned variations; if the cascade exited early (everyone
    # solved), the curve just holds its last value for the un-run variations.
    n_var = len(variations)
    cascade_curve = []
    last = 0.0
    for n in range(n_var):
        if n < len(per_variation_log):
            last = per_variation_log[n]["cumulative_solved"] / N if N else 0.0
        cascade_curve.append(last)
    x_cascade = list(range(1, n_var + 1))

    # K=1 base (greedy) accuracy: the first K=1 variation's own solved count.
    # (variation 0 is the K=1 base config in build_variations.)
    base_var_log = next((r for r in per_variation_log if r["K"] == 1), None)
    base_greedy_acc = (base_var_log["solved"] / N) if (base_var_log and N) else 0.0

    # Temperature best-of-n baseline (same n axis as the cascade by default).
    temp_bon_curve = None
    temp_per_task_first = None
    n_temp = None
    if not args.skip_temp_bon:
        n_temp = args.temp_bon_samples if args.temp_bon_samples is not None else n_var
        base_v = next((v for v in variations if v["K"] == 1), variations[0])
        est = N * n_temp
        print(f"\n[temp-bon] running temp={args.temp} best-of-n baseline: "
              f"N={N} items x n_temp={n_temp} samples = ~{est} generations "
              f"(base config K={base_v['K']} blocks=[{base_v['block_start']},{base_v['block_end']}])")
        temp_bon_curve, temp_per_task_first = run_temp_bon(
            model, processor, items, base_v,
            n_temp, args.temp, args.temp_bon_batch,
            args.max_new_tokens, input_device, verbose=args.verbose,
        )

    # ---- Print summary
    print("\n=== MCQ best-of-n summary ===")
    print(f"N = {N}")
    print(f"Final solved = {total_solved}/{N} = {final_acc*100:.2f}%")
    print(f"Unsolved     = {len(remaining)}")
    print(f"K=1 base (greedy) accuracy = {base_greedy_acc*100:.2f}%")
    if temp_bon_curve is not None:
        print(f"temp={args.temp} best-of-1   = {temp_bon_curve[0]*100:.2f}%")
        print(f"temp={args.temp} best-of-{n_temp}  = {temp_bon_curve[-1]*100:.2f}%")

    header = f"{'var':>4} | {'K':>3} | {'blocks':>9} | {'attempted':>9} | {'solved':>6} | {'cum':>6}"
    print(header)
    print("-" * len(header))
    for row in per_variation_log:
        print(f"{row['var_idx']:>4} | {row['K']:>3} | "
              f"[{row['block_start']:>2},{row['block_end']:>2}] | "
              f"{row['attempted']:>9} | {row['solved']:>6} | "
              f"{row['cumulative_solved']:>6}")

    print("\nPer-task:")
    for t in sorted(task_total.keys()):
        s, n = task_solved[t], task_total[t]
        print(f"  {t:<10}  {s}/{n}  ({(s/n*100 if n else 0):.1f}%)")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))

    # Cascade best-of-n
    plt.plot(x_cascade, [v * 100 for v in cascade_curve],
             marker="o", markersize=4, linewidth=1.8,
             label="Cascade best-of-n (K / block-window search)")

    # Temperature best-of-n baseline
    if temp_bon_curve is not None:
        x_temp = list(range(1, len(temp_bon_curve) + 1))
        plt.plot(x_temp, [v * 100 for v in temp_bon_curve],
                 marker="s", markersize=4, linewidth=1.8,
                 label=f"Base model temp={args.temp} best-of-n")

    # K=1 base greedy dashed line
    plt.axhline(base_greedy_acc * 100, linestyle="--", color="gray", linewidth=1.5,
                label=f"K=1 base (greedy) = {base_greedy_acc*100:.1f}%")

    plt.xlabel("n  (number of variations / samples tried)")
    plt.ylabel("Accuracy (%)")
    plt.title("Best-of-n accuracy vs n (MCQ)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plot_path = Path(args.plot_output)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[plot] wrote {plot_path}")

    # -------------------------------------------------------------------------
    # TXT log
    # -------------------------------------------------------------------------
    out_path = Path(args.output)
    with out_path.open("w") as f:
        f.write("MCQ Best-of-n Evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model:        {args.model}\n")
        if args.lora_dir:
            f.write(f"LoRA:         {args.lora_dir}\n")
        f.write(f"Dataset dir:  {args.dataset_dir}\n")
        f.write(f"N samples:    {N}\n")
        f.write(f"K_eval:       {args.K_eval}\n")
        f.write(f"block range:  [{args.block_min}, {args.block_max}]\n")
        f.write(f"window:       {args.block_window} (stride 1)\n")
        f.write(f"variations:   {len(variations)}\n")
        f.write(f"temp baseline: {'disabled' if args.skip_temp_bon else f'temp={args.temp}, n_temp={n_temp}'}\n\n")

        f.write("Per-variation results\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'var':>4} | {'K':>3} | {'blocks':>9} | "
                f"{'attempted':>9} | {'solved':>6} | {'cum_solved':>10} | "
                f"{'cum_acc%':>9}\n")
        for row in per_variation_log:
            cum_acc = row['cumulative_solved'] / N * 100 if N else 0.0
            f.write(f"{row['var_idx']:>4} | {row['K']:>3} | "
                    f"[{row['block_start']:>2},{row['block_end']:>2}] | "
                    f"{row['attempted']:>9} | {row['solved']:>6} | "
                    f"{row['cumulative_solved']:>10} | {cum_acc:>8.2f}%\n")

        # variations that were never reached (everyone solved already) — for transparency
        unrun = len(variations) - len(per_variation_log)
        if unrun > 0:
            f.write(f"\n({unrun} variation(s) not executed — cascade exited early)\n")

        f.write("\nPer-task breakdown\n")
        f.write("-" * 60 + "\n")
        for t in sorted(task_total.keys()):
            s, n = task_solved[t], task_total[t]
            f.write(f"  {t:<10}  {s}/{n}  ({(s/n*100 if n else 0):.2f}%)\n")

        f.write("\nPer-variation x per-task (samples newly solved)\n")
        f.write("-" * 60 + "\n")
        tasks_sorted = sorted(task_total.keys())
        f.write(f"{'var':>4} | {'K':>3} | {'blocks':>9} | "
                + " | ".join(f"{t:>10}" for t in tasks_sorted) + "\n")
        for row in per_variation_log:
            v_idx = row['var_idx']
            line = (f"{v_idx:>4} | {row['K']:>3} | "
                    f"[{row['block_start']:>2},{row['block_end']:>2}] | "
                    + " | ".join(f"{var_task[v_idx].get(t,0):>10}" for t in tasks_sorted))
            f.write(line + "\n")

        # ---- Best-of-n curve data
        f.write("\nBest-of-n curves (accuracy %)\n")
        f.write("-" * 60 + "\n")
        f.write(f"K=1 base (greedy) accuracy: {base_greedy_acc*100:.2f}%\n\n")
        if temp_bon_curve is not None:
            f.write(f"{'n':>4} | {'cascade_bon%':>13} | {'temp'+str(args.temp)+'_bon%':>14}\n")
            for i in range(n_var):
                casc = cascade_curve[i] * 100
                tval = (temp_bon_curve[i] * 100) if i < len(temp_bon_curve) else float('nan')
                tstr = f"{tval:13.2f}" if tval == tval else f"{'n/a':>13}"
                f.write(f"{i+1:>4} | {casc:13.2f} | {tstr}\n")
            # if temp baseline has more n than cascade variations, dump the tail too
            if len(temp_bon_curve) > n_var:
                for i in range(n_var, len(temp_bon_curve)):
                    f.write(f"{i+1:>4} | {'n/a':>13} | {temp_bon_curve[i]*100:13.2f}\n")
        else:
            f.write(f"{'n':>4} | {'cascade_bon%':>13}\n")
            for i in range(n_var):
                f.write(f"{i+1:>4} | {cascade_curve[i]*100:13.2f}\n")

        f.write(f"\nPlot saved to: {plot_path}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"FINAL: {total_solved}/{N} = {final_acc*100:.2f}%\n")
        f.write(f"Unsolved: {len(remaining)}\n")

    print(f"[log] wrote {out_path}")


if __name__ == "__main__":
    main()