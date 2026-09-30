# RVD Best-of-n VLM Evaluation

Three evaluation scripts that run a **best-of-n (K / block-window) search** on a
vision-language model and compare it against a **temperature best-of-n baseline**.
The model is never reloaded between variations — the RVD backend is re-patched in
place with a new `(K, block_start, block_end)`.

Each script produces a per-variation log, a final accuracy/closeness number, a
`.txt` report, and a best-of-n curve `.png`.

| Script | Task type | Datasets it fits | Scoring |
|---|---|---|---|
| `count_bon.py` | open-ended **numeric** | CountBenchQA, and similar count/number datasets | continuous closeness in [0,1] (`relerr` / `ratio` / `abs`) |
| `mcq_bon.py` | **multiple choice** | **CV-Bench, AI2D, MMMU**, any MCQ dataset | binary (letter match) |
| `realworldqa_bon.py` | **verifiable** short answer | RealWorldQA | binary (rule-based verifier: letter / numeric / text) |

> `mcq_bon.py` is dataset-agnostic: it only needs an image, a question, a list of
> choices, and a letter/text answer. Point `--dataset_dir` at any MCQ dataset saved
> with `load_from_disk` (CV-Bench, AI2D, MMMU, …). The per-task breakdown will just
> show `Unknown` for datasets without a `task` column, which is harmless.

---

## 1. Pick the RVD backend (one line, top of each script)

Every script imports the RVD implementation that matches your model. Uncomment
**exactly one** line:

```python
# import rvd_qwen_2_5_evo as rvd     # Qwen2.5-VL
import rvd_qwen_3_evo   as rvd   # Qwen3-VL
# import rvd_qwen_3_5_evo as rvd   # Qwen3.5-VL  (needs transformers 5.16.1)
```

The scripts ship with **Qwen3-VL** active because that matches the default
environment below. To evaluate a different model, comment the current line and
uncomment the one you want.

---

## 2. Environment

Tested with **torch 2.5.1+cu121**.

```
transformers 4.57.0   ->  Qwen2.5-VL, Qwen3-VL   (default environment)
transformers 5.16.1   ->  Qwen3.5-VL             (same env, upgrade only)
```

### Default env (Qwen2.5 / Qwen3)

Run the helper script:

```bash
bash setup_env.sh
```

or the same steps by hand:

```bash
# create + activate an env (conda or venv), then:
pip install --upgrade pip
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.57.0"
pip install "accelerate>=0.34" datasets pillow matplotlib
pip install peft            
```

### Qwen3.5-VL (upgrade only — no new env)

Keep the **same** environment and just bump transformers:

```bash
pip install -U "transformers==5.16.1"
```

Then switch the import at the top of the script to `rvd_qwen_3_5_evo`.

> **Qwen3.5-VL note — `enable_thinking=False`.** Qwen3.5 has a *thinking* mode.
> For these short-answer evals you almost always want it **off**. Each script marks
> the exact spot in `_build_inputs` (the `processor.apply_chat_template(...)` call):
>
> ```python
> text = processor.apply_chat_template(
>     messages, tokenize=False, add_generation_prompt=True,
>     enable_thinking=False,   # <-- Qwen3.5-VL only; leave out for Qwen2.5 / Qwen3
> )
> ```
>
> Add `enable_thinking=False,` **only** when running Qwen3.5. It is a Qwen3.5 flag
> and will error on Qwen2.5 / Qwen3, so leave it commented for those models.

---

## 3. Running

Common flags across all three scripts:

- `--model` — path to the model weights
- `--dataset_dir` — `load_from_disk` folder
- `--max_samples` — cap the number of eval samples
- `--K_eval 1 2 4` — K values tried in order (smallest first)
- `--block_min / --block_max / --block_window` — sliding block-window search
- `--temp` — temperature for the baseline curve (default 0.6)
- `--skip_temp_bon` — skip the (expensive) temperature baseline
- `--output` / `--plot_output` — report `.txt` and curve `.png`
- `--lora_dir` — optional PEFT LoRA adapter
- `--verbose` — per-sample logging

### MCQ (CV-Bench / AI2D / MMMU)

```bash
python mcq_bon.py \
    --model /path/to/Qwen3-VL-8B-Instruct  \
    --dataset_dir /path/to/CV-Bench \
    --max_samples 5000 \
    --K_eval 1 2 4 \
    --block_min 0 --block_max 5 --block_window 3 --temp 0.6 \
    --output mcq_bon_results.txt \
    --plot_output mcq_bon_curve.png
```

Swap `--dataset_dir` to your AI2D or MMMU folder to run those instead.

### Open-ended numeric (CountBenchQA)

```bash
python count_bon.py \
    --model /path/to/Qwen3-VL-8B-Instruct  \
    --dataset_dir /path/to/CountBenchQA \
    --max_samples 5000 \
    --K_eval 1 2 4 \
    --block_min 0 --block_max 5 --block_window 3 \
    --score_metric relerr \
    --output count_bon_results.txt \
    --plot_output count_bon_curve.png
```

`--score_metric`: `relerr` (default, `1 - |pred-gt|/|gt|`), `ratio`, or `abs` (exact).

### RealWorldQA (verifiable)

```bash
python realworldqa_bon.py \
    --model /path/to/Qwen3-VL-8B-Instruct \
    --dataset_dir /path/to/RealWorldQA \
    --max_samples 5000 \
    --K_eval 1 2 4 --block_min 0 --block_max 5 --block_window 3 \
    --max_new_tokens 256 \
    --output realworldqa_bon_results.txt \
    --plot_output realworldqa_bon_curve.png
```

---

## 4. Switching model — quick checklist

1. Edit the import block at the top of the script (section 1).
2. If moving **to** Qwen3.5: `pip install -U "transformers==5.16.1"` and add
   `enable_thinking=False,` in `_build_inputs`.
3. If moving **back** to Qwen2.5 / Qwen3: `pip install "transformers==4.57.0"` and
   remove `enable_thinking=False,` again.
4. Point `--model` at the matching weights.

---

## 5. Files

```
count_bon.py         open-ended numeric best-of-n
mcq_bon.py           multiple-choice best-of-n (CV-Bench / AI2D / MMMU / …)
realworldqa_bon.py   RealWorldQA verifiable best-of-n
setup_env.sh         default environment installer (torch 2.5.1+cu121, transformers 4.57.0)
README.md            this file
```

Requires your RVD backends on the import path: `rvd_qwen_2_5_evo.py`,
`rvd_qwen_3_evo.py`, `rvd_qwen_3_5_evo.py` (each must expose `patch_model`,
`set_K`, and optionally `unpatch_model`).
