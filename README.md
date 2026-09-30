# Architectural Sampling: Test-Time Scaling via Computational Diversity in Frozen Vision-Language Models

## 1. Introduction

<!-- FIGURE: method overview. Drop the image at figures/method_overview.png -->
<p align="center">
  <img src="figures/method_overview.png" alt="Method overview: RVD best-of-n vs temperature baseline" width="720"/>
  <br/>
  <em>Figure 1. Method overview. (placeholder — add figures/method_overview.png)</em>
</p>

**Method.** Instead of a single greedy decode, we generate several candidate
answers per (image, question) pair and count a sample as *solved* the first time
any candidate is correct. Candidates are produced by an **RVD backend**
(`rvd_qwen_*_evo`) that re-patches the model **in place** with a search
configuration `(K, block_start, block_end)` — the model is never reloaded between
variations. We sweep `K` together with a sliding **block-window** and plot the
resulting **best-of-n** curve against a **temperature best-of-n** baseline
(unbiased pass@k). Because both use the *same* deterministic verifier, the
search-vs-temperature comparison is apples-to-apples.

We evaluate three answer types with one eval script each:

| Answer type | Eval script | Datasets |
|---|---|---|
| Open-ended **numeric** | `count_bon.py` | CountBenchQA, CountQA |
| **Multiple choice** | `mcq_bon.py` | CV-Bench, MMStar, AI2D, ScienceQA, A-OKVQA, MMMU, BLINK |
| **Verifiable** free-form | `realworldqa_bon.py` | RealWorldQA |

The model is selected by which RVD backend the script imports, so the same code
runs on Qwen2.5-VL, Qwen3-VL and Qwen3.5-VL.

---

## 2. Environment

Tested with **torch 2.5.1+cu121**.

```
transformers 4.57.0   ->  Qwen2.5-VL, Qwen3-VL   (default environment)
transformers 5.16.1   ->  Qwen3.5-VL             (same env, upgrade only)
```

Create the environment, activate it, then install everything with the helper
script:

```bash
# 1. create the environment (Python 3.10)
conda create -y -n rvd python=3.10

# 2. activate it — source the conda hook first so `conda activate` works
#    in any shell (this avoids the "Run 'conda init' before 'conda activate'" error)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rvd

# 3. install torch 2.5.1+cu121, transformers 4.57.0 and the other deps
bash setup_env.sh
```

`setup_env.sh` installs into the **currently active** environment (torch,
transformers 4.57.0, accelerate, datasets, pillow, matplotlib, peft) and prints
a torch/CUDA/transformers sanity check at the end.

### Qwen3.5-VL (upgrade only — no new env)

Keep the same environment and bump transformers:

```bash
pip install -U "transformers==5.16.1"
```

> **Qwen3.5-VL note — `enable_thinking=False`.** Qwen3.5 has a *thinking* mode; for
> these short-answer evals you want it **off**. Each eval script marks the exact
> spot in `_build_inputs` (the `processor.apply_chat_template(...)` call). Add
> `enable_thinking=False,` there **only** for Qwen3.5 — it is a Qwen3.5 flag and
> will error on Qwen2.5 / Qwen3, so leave it out for those models.

### Selecting the model (top of each eval script)

Uncomment **exactly one** RVD backend:

```python
import rvd_qwen_2_5_evo as rvd     # Qwen2.5-VL
# import rvd_qwen_3_evo   as rvd   # Qwen3-VL
# import rvd_qwen_3_5_evo as rvd   # Qwen3.5-VL  (needs transformers 5.16.1)
```

---

## 3. Data

Download and normalize every benchmark **once, on a machine with internet** (e.g.
a login node), then copy the output folders to the cluster. `download_mcq_datasets.py`
writes each dataset with `save_to_disk`, in the exact schema its eval script reads:

```bash
# everything into ./vlm_data/<name>
python download_mcq_datasets.py --all --out_root ./vlm_data

# or a subset (any mix of numeric / MCQ / verifiable)
python download_mcq_datasets.py --datasets cvbench countbench realworldqa --out_root ./vlm_data

# cap samples per dataset while downloading
python download_mcq_datasets.py --datasets ai2d --limit 2638 --out_root ./vlm_data
```

Schemas written (one per eval script):

| Group | For | Columns |
|---|---|---|
| MCQ | `mcq_bon.py` | `image, question, choices, answer(letter), task` |
| Numeric | `count_bon.py` | `image, question, answer(int)` |
| Verifiable | `realworldqa_bon.py` | `image, question, answer(verbatim)` |

> **HuggingFace repo ids.** The ids/splits for `countbench`, `countqa` and
> `realworldqa` are set in the `NUMERIC_SOURCES` / `REALWORLDQA_REPO` blocks near
> the bottom of `download_mcq_datasets.py`. `countqa` defaults to the CountBenchQA
> repo so `--all` runs out of the box — point it at your own CountQA source if it
> is a different dataset. If a repo/column differs in your account, that dataset
> prints `[ERROR] <name> failed: ...` and the others still download.

---

## 4. Experiments

Each eval loads a model once, then for every search variation re-patches the RVD
backend with a new `(K, block_start, block_end)`, decodes, and scores. It writes a
`.txt` report and a best-of-n curve `.png`.

Shared flags: `--model`, `--dataset_dir`, `--max_samples`, `--K_eval`,
`--block_min/--block_max/--block_window`, `--temp` (baseline temperature),
`--skip_temp_bon`, `--output`, `--plot_output`, `--lora_dir`, `--verbose`.

### 4.1 Multiple choice (CV-Bench / AI2D / MMMU / …)

```bash
python mcq_bon.py \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --dataset_dir ./vlm_data/cvbench \
    --max_samples 2638 \
    --K_eval 1 2 4 \
    --block_min 0 --block_max 5 --block_window 3 --temp 0.6 \
    --output results/mcq_cvbench_bon.txt \
    --plot_output results/mcq_cvbench_bon.png
```

Swap `--dataset_dir` to `./vlm_data/ai2d`, `./vlm_data/mmmu`, etc. to run the
other MCQ sets with the same command.

### 4.2 Open-ended numeric (CountBenchQA / CountQA)

```bash
python count_bon.py \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --dataset_dir ./vlm_data/countbench \
    --max_samples 491 \
    --K_eval 1 2 4 \
    --block_min 0 --block_max 4 --block_window 3 \
    --score_metric relerr \
    --output results/count_countbench_bon.txt \
    --plot_output results/count_countbench_bon.png
```

`--score_metric`: `relerr` (default), `ratio`, or `abs` (exact match).

### 4.3 Verifiable free-form (RealWorldQA)

```bash
python realworldqa_bon.py \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --dataset_dir ./vlm_data/realworldqa \
    --max_samples 765 \
    --K_eval 1 2 4 --block_min 0 --block_max 5 --block_window 3 \
    --max_new_tokens 256 \
    --output results/realworldqa_bon.txt \
    --plot_output results/realworldqa_bon.png
```

### Switching model — checklist

1. Edit the RVD import at the top of the script (Section 2).
2. Moving **to** Qwen3.5: `pip install -U "transformers==5.16.1"` and add
   `enable_thinking=False,` in `_build_inputs`.
3. Moving **back** to Qwen2.5 / Qwen3: `pip install "transformers==4.57.0"` and
   remove `enable_thinking=False,`.
4. Point `--model` at the matching weights.

---

## 5. Results

<!-- FIGURE: best-of-n curves. Drop the image at figures/results_curves.png -->
<p align="center">
  <img src="figures/results_curves.png" alt="Best-of-n vs temperature best-of-n accuracy" width="720"/>
  <br/>
  <em>Figure 2. Best-of-n (search) vs temperature baseline. (placeholder — add figures/results_curves.png)</em>
</p>

All runs write their `.txt` reports and `.png` curves to the **`results/`**
folder. **The collected results are already in `results/` — open that folder to
see the per-dataset logs and best-of-n curves.**

---

## 6. Repository layout

```
count_bon.py               open-ended numeric best-of-n            -> CountBenchQA / CountQA
mcq_bon.py                 multiple-choice best-of-n               -> CV-Bench / AI2D / MMMU / ...
realworldqa_bon.py         RealWorldQA verifiable best-of-n
download_mcq_datasets.py   one-shot dataset downloader (all schemas)
setup_env.sh               environment installer (torch 2.5.1+cu121, transformers 4.57.0)
README.md                  this file
figures/                   figures referenced above (method_overview.png, results_curves.png)
results/                   .txt reports and .png curves from the runs
```

Requires your RVD backends on the import path: `rvd_qwen_2_5_evo.py`,
`rvd_qwen_3_evo.py`, `rvd_qwen_3_5_evo.py` (each exposing `patch_model`, `set_K`,
and optionally `unpatch_model`).
