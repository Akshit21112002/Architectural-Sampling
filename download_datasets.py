#!/usr/bin/env python
"""
Download several multimodal (VLM) benchmarks and normalize each into the schema
that the matching eval script reads, then save_to_disk so they can be loaded
fully offline with load_from_disk.

Each dataset is saved for exactly ONE eval script:

  MCQ  -> mcq_bon.py         cvbench, mmstar, ai2d, scienceqa, aokvqa, mmmu, blink
    columns: image, question, choices, answer, task
      choices : list[str] of options ([] if options are inside the question text)
      answer  : a single letter "A".."H"
      task    : string used for the per-task breakdown

  NUM  -> count_bon.py       countbench, countqa
    columns: image, question, answer
      answer  : an integer count (count_bon.py reads it via 'answer'/'number'/...)

  VER  -> realworldqa_bon.py realworldqa
    columns: image, question, answer
      answer  : the raw short answer kept VERBATIM (a letter, word, or number);
                realworldqa_bon.py's verifier decides how to grade it, and it
                parses any embedded A./B./C. choices out of the question text.

Run this on a machine WITH internet (e.g. a login node), then copy the output
folders to the cluster and point --dataset_dir at them.

Requires: pip install -U datasets pillow huggingface_hub

Examples
--------
# everything, into ./vlm_data/<name>
python download_datasets.py --all --out_root ./vlm_data

# a few of them (any mix of MCQ / numeric / verifiable)
python download_datasets.py --datasets cvbench countbench realworldqa --out_root ./vlm_data

# cap a big one while downloading
python download_datasets.py --datasets ai2d --limit 2638 --out_root ./vlm_data

NOTE: the HuggingFace repo ids for countbench / countqa / realworldqa are set in
the SOURCES dicts below. If your account mirrors them under different ids or
splits, edit those two dicts — the schema handling stays the same.
"""

import argparse
import ast
import re
from pathlib import Path

from datasets import load_dataset, Dataset
from datasets import Image as HFImage

_LETTERS = "ABCDEFGH"


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _to_letter(ans, choices):
    """Best-effort convert an answer (letter / '(A)' / 'A.' / index / text) -> 'A'..'H'."""
    if ans is None:
        return None
    s = str(ans).strip()
    # already a bare letter
    if len(s) == 1 and s.upper() in _LETTERS:
        return s.upper()
    # "(A)" / "A." / "A)" style: a letter wrapped in / followed by punctuation
    # (anchored so a short text answer like "cat" is NOT read as "A").
    m = re.match(r"^\(?([A-Ha-h])[\.\)\s]", s) or re.match(r"^\(?([A-Ha-h])\)?$", s)
    if m:
        return m.group(1).upper()
    # integer index into choices (0-based)
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(_LETTERS):
            return _LETTERS[i]
    # match against choice text
    if isinstance(choices, (list, tuple)):
        for i, c in enumerate(choices):
            if str(c).strip().lower() == s.lower():
                return _LETTERS[i]
    return None


def _first(ex, keys, default=None):
    """Return the first present, non-None value among `keys`."""
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return default


def _save(records, out_dir):
    """records: list of dicts; must have 'image' + 'answer', plus 'question'.
    'choices'/'task' are optional (MCQ only)."""
    records = [r for r in records if r.get("answer") is not None and r.get("image") is not None]
    if not records:
        raise RuntimeError(f"No usable records for {out_dir}")
    ds = Dataset.from_list(records)
    ds = ds.cast_column("image", HFImage())      # keep as decodable image feature
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"[saved] {out_dir}  ({len(ds)} samples)")
    # quick sanity peek (works for all three schemas)
    ex = ds[0]
    bits = [f"answer={ex['answer']!r}"]
    if "task" in ex:
        bits.append(f"task={ex['task']!r}")
    if "choices" in ex:
        bits.append(f"n_choices={len(ex['choices'])}")
    bits.append(f"q[:60]={str(ex.get('question', ''))[:60]!r}")
    print("        e.g. " + " ".join(bits))


def _take(iterable, limit):
    return iterable if limit is None else iterable.select(range(min(limit, len(iterable))))


def _load_any_split(repo, prefer=("test", "validation", "val", "train")):
    """load_dataset(repo) and pick a sensible split without guessing wrong."""
    dd = load_dataset(repo)
    if hasattr(dd, "keys"):          # DatasetDict
        keys = list(dd.keys())
        for s in prefer:
            if s in keys:
                return dd[s]
        return dd[keys[0]]
    return dd                        # already a single Dataset


# -----------------------------------------------------------------------------
# per-dataset builders  (return list[record])
# -----------------------------------------------------------------------------

def build_cvbench(limit):
    # nyu-visionx/CV-Bench: already (image, question, choices, answer, task).
    # answer is like "(A)"; single 'test' split of 2638 items.
    ds = _take(load_dataset("nyu-visionx/CV-Bench", split="test"), limit)
    out = []
    for ex in ds:
        choices = list(ex.get("choices") or [])
        out.append({
            "image":    ex["image"],
            "question": str(ex["question"]).strip(),
            "choices":  [str(c) for c in choices],
            "answer":   _to_letter(ex.get("answer"), choices),
            "task":     str(ex.get("task") or "cvbench"),
        })
    return out


def build_mmstar(limit):
    # options are already embedded in `question`; answer is a letter.
    ds = _take(load_dataset("Lin-Chen/MMStar", split="val"), limit)
    out = []
    for ex in ds:
        out.append({
            "image":    ex["image"],
            "question": str(ex["question"]).strip(),
            "choices":  [],  # already inside the question
            "answer":   _to_letter(ex.get("answer"), []),
            "task":     str(ex.get("category", "mmstar")),
        })
    return out


def build_ai2d(limit):
    # question + options(list of 4) + answer(index string). image is the diagram.
    ds = _take(load_dataset("lmms-lab/ai2d", split="test"), limit)
    out = []
    for ex in ds:
        choices = list(ex.get("options") or [])
        out.append({
            "image":    ex["image"],
            "question": str(ex["question"]).strip(),
            "choices":  [str(c) for c in choices],
            "answer":   _to_letter(ex.get("answer"), choices),
            "task":     "ai2d",
        })
    return out


def build_scienceqa(limit):
    # image subset only; choices is a list, answer is an int index.
    ds = _take(load_dataset("lmms-lab/ScienceQA-IMG", split="test"), limit)
    out = []
    for ex in ds:
        if ex.get("image") is None:      # be safe even though this is the IMG subset
            continue
        choices = list(ex.get("choices") or [])
        out.append({
            "image":    ex["image"],
            "question": str(ex["question"]).strip(),
            "choices":  [str(c) for c in choices],
            "answer":   _to_letter(ex.get("answer"), choices),
            "task":     str(ex.get("subject") or "scienceqa"),
        })
    return out


def build_aokvqa(limit):
    # MC form: choices(list of 4) + correct_choice_idx(int).
    ds = _take(load_dataset("HuggingFaceM4/A-OKVQA", split="validation"), limit)
    out = []
    for ex in ds:
        choices = list(ex.get("choices") or [])
        idx = ex.get("correct_choice_idx")
        out.append({
            "image":    ex["image"],
            "question": str(ex["question"]).strip(),
            "choices":  [str(c) for c in choices],
            "answer":   _to_letter(idx, choices),
            "task":     "aokvqa",
        })
    return out


def build_mmmu(limit):
    # options is a stringified python list; answer is a letter; may be multi-image
    # with "<image N>" tokens. Keep only single-image multiple-choice items.
    ds = load_dataset("lmms-lab/MMMU", split="validation")
    out = []
    for ex in ds:
        if str(ex.get("question_type", "")).lower() not in ("multiple-choice", "multiple_choice", ""):
            continue
        img = ex.get("image_1")
        if img is None:
            continue
        if ex.get("image_2") is not None:     # skip genuine multi-image questions
            continue
        raw_opts = ex.get("options")
        try:
            choices = ast.literal_eval(raw_opts) if isinstance(raw_opts, str) else list(raw_opts or [])
        except Exception:
            continue
        if not choices:
            continue
        q = str(ex.get("question", ""))
        for i in range(1, 8):                 # strip "<image i>" placeholders
            q = q.replace(f"<image {i}>", "").strip()
        out.append({
            "image":    img,
            "question": q,
            "choices":  [str(c) for c in choices],
            "answer":   _to_letter(ex.get("answer"), choices),
            "task":     str(ex.get("subfield") or ex.get("subject") or "mmmu"),
        })
        if limit is not None and len(out) >= limit:
            break
    return out


_BLINK_CONFIGS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]


def build_blink(limit):
    # BLINK-Benchmark/BLINK: one config per sub-task; keep single-image MCQ items.
    out = []
    for cfg in _BLINK_CONFIGS:
        try:
            ds = load_dataset("BLINK-Benchmark/BLINK", cfg, split="val")
        except Exception as e:
            print(f"  [blink] skip {cfg}: {e}")
            continue
        for ex in ds:
            if ex.get("image_2") is not None:      # keep single-image only
                continue
            img = ex.get("image_1")
            choices = list(ex.get("choices") or [])
            if img is None or not choices:
                continue
            out.append({
                "image":    img,
                "question": str(ex.get("question", "")).strip(),
                "choices":  [str(c) for c in choices],
                "answer":   _to_letter(ex.get("answer"), choices),
                "task":     str(ex.get("sub_task") or cfg),
            })
            if limit is not None and len(out) >= limit:
                return out
    return out


# -----------------------------------------------------------------------------
# NUMERIC datasets  -> count_bon.py   (schema: image, question, answer=int)
# -----------------------------------------------------------------------------

# HuggingFace repo id per numeric dataset. Edit if your mirror differs.
NUMERIC_SOURCES = {
    "countbench": "vikhyatk/CountBenchQA",
    "countqa":    "Jayant-Sravan/CountQA",
}


def _make_numeric_builder(repo):
    def _build(limit):
        ds = _take(_load_any_split(repo), limit)
        out = []
        for ex in ds:
            num = _first(ex, ["number", "answer", "answers", "count", "gt_answer", "label"])
            if isinstance(num, list) and num:
                num = num[0]
            try:
                num = int(num)                 # keep the answer column a clean int
            except (TypeError, ValueError):
                num = None                     # dropped by _save (count needs a number)
            out.append({
                "image":    _first(ex, ["image", "Image", "img", "decoded_image"]),
                "question": str(_first(ex, ["question", "text", "query", "Question"], "")).strip(),
                "answer":   num,
            })
        return out
    return _build


# -----------------------------------------------------------------------------
# VERIFIABLE dataset  -> realworldqa_bon.py  (schema: image, question, answer=str)
# -----------------------------------------------------------------------------

REALWORLDQA_REPO = "lmms-lab/RealWorldQA"


def build_realworldqa(limit):
    # Keep every item and store the answer VERBATIM — realworldqa_bon.py's
    # verifier handles letters, numbers, and free text, and parses any embedded
    # A./B./C. choices out of the question itself.
    ds = _take(_load_any_split(REALWORLDQA_REPO), limit)
    out = []
    for ex in ds:
        ans = _first(ex, ["answer", "Answer", "gt_answer", "label"])
        out.append({
            "image":    _first(ex, ["image", "Image", "img", "decoded_image"]),
            "question": str(_first(ex, ["question", "Question", "query"], "")).strip(),
            "answer":   None if ans is None else str(ans).strip(),
        })
    return out


BUILDERS = {
    # MCQ -> mcq_bon.py
    "cvbench":    build_cvbench,
    "mmstar":     build_mmstar,
    "ai2d":       build_ai2d,
    "scienceqa":  build_scienceqa,
    "aokvqa":     build_aokvqa,
    "mmmu":       build_mmmu,
    "blink":      build_blink,
    # verifiable -> realworldqa_bon.py
    "realworldqa": build_realworldqa,
}

# numeric -> count_bon.py  (registered from NUMERIC_SOURCES)
for _name, _repo in NUMERIC_SOURCES.items():
    BUILDERS[_name] = _make_numeric_builder(_repo)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="./vlm_data")
    p.add_argument("--datasets", nargs="+", choices=list(BUILDERS.keys()))
    p.add_argument("--all", action="store_true",
                   help="Download all datasets: " + ", ".join(BUILDERS.keys()))
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on samples per dataset (applied at download).")
    args = p.parse_args()

    if args.all:
        names = list(BUILDERS.keys())
    elif args.datasets:
        names = args.datasets
    else:
        p.error("pass --all or --datasets ...")

    for name in names:
        print(f"\n==== {name} ====")
        try:
            records = BUILDERS[name](args.limit)
            n_bad = sum(1 for r in records if r["answer"] is None)
            if n_bad:
                print(f"[warn] {n_bad} records had unresolvable answers (dropped on save)")
            _save(records, Path(args.out_root) / name)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")


if __name__ == "__main__":
    main()
