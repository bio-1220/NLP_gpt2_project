# GPT-2 Implementation and Emotion-Conditioned Bilingual Poetry Generation

*CSEG321 Final Project — Jihong Park, Jeewu Lee, Hyeongwon Jo (Sogang University)*

A from-scratch GPT-2 (English, ~124M) and KoGPT2 (Korean) project that:

1. **Base tasks** — runs three standard downstream tasks: sentiment classification, paraphrase detection, and sonnet generation.
2. **Final experiment** — tests whether providing a poem's full ground-truth emotion curve (including future lines) improves poem continuation.

> **Main finding:** Providing the complete oracle emotion curve at both training and inference time — across 3 English sentiment scales and Korean 44-/6-class label resolutions — did **not** beat the unconditioned baseline on continuation quality (chrF / BERTScore). The oracle curve was indistinguishable from a shuffled curve (no content signal), and in Korean the model given the oracle emotion sequence actually followed the emotion flow *less* well. The only reproducible effect was improved Volta (emotional turning-point) placement from the train-average curve — a structural prior, not poem-specific content.

---

## Setup

| Item | Value |
|---|---|
| Conda env | `NLP_Project` |
| PyTorch | 2.11.0+cu128 (verified on RTX 5070 Ti 17GB; cu118 lacks sm_120 support) |
| Key packages | transformers 4.46.3, sacrebleu, bert-score, pandas, pyarrow, einops, scikit-learn |

**Windows note:** `conda run` crashes on Korean output (cp949). Call the env's Python directly:

```bash
PYTHONIOENCODING=utf-8 "C:/Users/<user>/anaconda3/envs/NLP_Project/python.exe" <script> ...
```

Below this is written simply as `python`. **Run all scripts from the repo root.**

---

## How to Run

### 1. Data preparation (once, after clone)

```bash
python scripts/prepare_external_data.py            # emotion (en/ko) + poem (ko) base JSONL
python scripts/smoke_test_data_bridge.py           # validation (16000/11880/368, chrF identity 100)

# --- final-experiment data ---
python scripts/prepare_shakespeare_sentiment_trajectory.py --continuous                          # EN curve m
python scripts/prepare_shakespeare_sentiment_trajectory.py --continuous --score_col sentiment_v  # EN curve v
python scripts/prepare_shakespeare_sentiment_trajectory.py --continuous --score_col sentiment_r  # EN curve r
python scripts/prepare_round3_en.py                # exclude dev#138 + shuffled control + plain base
python scripts/prepare_kpoem_fulltraj_fine.py --label_mode fine   # KO full-line 44-class labels
python scripts/prepare_kpoem_fulltraj_fine.py --label_mode six    # KO full-line 6-class labels
```

### 2. Base tasks

```bash
python sanity_check.py                                        # GPT-2 implementation check
python classifier.py --use_gpu --fine-tune-mode full-model    # SST + CFIMDB sentiment classification
python paraphrase_detection.py --use_gpu                      # Quora paraphrase detection
python sonnet_generation.py --use_gpu                         # sonnet generation
python analysis/build_report.py                               # base-task report PDF
```

### 3. Final experiment

```bash
# English: base 1 + (oracle/average/shuffled x m/v/r) 9 = 10 models, train + eval
bash scripts/run_round3.sh

# Korean: train full-line-label model (fine shown; for six, swap fine->six in filenames)
python training/train_poem.py --lang ko \
  --train_file data/processed/poem_ko_train_fulltraj_fine_r30.jsonl \
  --dev_file   data/processed/poem_ko_dev_fulltraj_fine_r30.jsonl \
  --setup_id R3K_fine --use_gpu --epochs 10 --batch_size 8 --max_length 768 \
  --out checkpoints/round3_curve/poem_ko_R3K_fine.pt
python training/evaluate_generation.py --lang ko \
  --ckpt checkpoints/round3_curve/poem_ko_R3K_fine.pt --setup_id R3K_fine \
  --eval_file data/processed/poem_ko_test_fulltraj_fine_r30_sowol.jsonl --use_gpu --n_samples 3

# Korean baseline re-eval (no training — reuse unconditioned model)
python training/evaluate_generation.py --lang ko \
  --ckpt checkpoints/round1_emotion_prefix/poem_ko_K0.pt --setup_id R3K_base \
  --eval_file data/processed/poem_ko_test_plain_r30_sowol.jsonl --use_gpu --n_samples 3

# Full-metric scoring (repeat per setup; --lang en/ko)
python training/score_generations.py --gen_file results/gen_R3_orc_m.jsonl --lang en --use_gpu

# Aggregate + report
python analysis/analyze_round3.py
python analysis/make_report_round3.py
```

---

## Results

### Base tasks

Common settings: 10 epochs, custom AdamW, best-dev checkpoint, seed 11711. Scores are on **dev** (true test labels are private).

| Task | Dataset | Mode | lr | batch | dev acc | macro-F1 |
|---|---|---|---|---|---|---|
| Sentiment | SST (5-class) | full | 1e-5 | 64 | **0.514** | 0.496 |
| Sentiment | SST (5-class) | last-linear | 1e-3 | 64 | 0.445 | 0.400 |
| Sentiment | CFIMDB (2-class) | full | 1e-5 | 8 | **0.971** | 0.971 |
| Sentiment | CFIMDB (2-class) | last-linear | 1e-3 | 8 | 0.857 | 0.856 |
| Paraphrase | Quora (283k) | full | 1e-5 | 32 | **0.896** | 0.889 |
| Sonnet gen | Shakespeare (131 poems) | full | 1e-5 | 8 | chrF 27.7–27.8 | — |

Full fine-tuning beats last-linear on both classifiers (+0.069 / +0.114), which is why full fine-tuning is fixed for all later experiments. 

### Final experiment

Setups (13 total): EN `base`, EN `oracle/average/shuffled × {m, v, r}` (9), KO `base`, KO `full-line 44-class`, KO `full-line 6-class`. EN dev#138 excluded (duplicate of private test #155), so EN eval n=11; KO eval = 11 Kim Sowol poems, 30%-of-length prefix.

Shared hyperparameters: 10 epochs / batch 8 / lr 1e-5 / full fine-tuning / max_length EN 384, KO 768; nucleus decoding (temp 0.8, top_p 0.9), 3 samples/poem, max_new_tokens 200.

Conditions: `Base` (poem text only), `Oracle` (poem's gold full trajectory), `Average` (mean trajectory over training poems — a genre-level structural prior), `Shuffled` (another poem's real trajectory under a cyclic permutation — preserves distribution/smoothness, destroys poem↔trajectory correspondence). Metrics: chrF, BERTScore F1, Volta distance (VDist, lower is better), trajectory correlation (Corr), Distinct-2.

**English sonnets (n=11, 3 samples/poem)**

| Condition | chrF | BERT-F1 | VDist ↓ | Corr | Dist-2 |
|---|---|---|---|---|---|
| **Base** | **27.69** | **0.833** | 4.58 | −0.020 | 0.840 |
| Oracle m | 25.29 | 0.827 | 4.76 | 0.019 | 0.803 |
| Average m | 24.88 | 0.826 | 3.52 | 0.012 | 0.775 |
| Shuffled m | 27.08 | 0.827 | 3.97 | 0.024 | 0.805 |
| Oracle v | 25.39 | 0.827 | 4.52 | −0.146 | 0.797 |
| Average v | 25.86 | 0.828 | **2.94** | −0.028 | 0.792 |
| Shuffled v | 26.14 | 0.826 | 5.12 | 0.057 | 0.785 |
| Oracle r | 26.50 | 0.830 | 4.27 | 0.058 | 0.808 |
| Average r | 26.48 | 0.827 | 4.58 | −0.090 | 0.786 |
| Shuffled r | 23.87 | 0.824 | 3.70 | 0.108 | 0.767 |

Base ranks first in both chrF and BERTScore. Oracle m/v cut per-poem chrF by 2.40 / 2.30 (worse on 10 of 11 poems); oracle r by 1.19 (worse on 8 of 11). Oracle curves are *not* systematically better than content-destroying shuffled controls, and the direction flips across sentiment instruments (m/v/r). The only repeated positive effect is structural: the average curve lowers Volta distance (2.94 / 3.52 vs 4.58) — a learnable genre prior for where a sonnet turns.

**Korean (11 Kim Sowol poems, 30%-of-length line input)**

| Condition | chrF | BERT-F1 | VDist ↓ | Corr | Agree |
|---|---|---|---|---|---|
| Base | 6.28 | 0.612 | 2.30 | **0.156** | **0.326** |
| Oracle 44-class | 6.50 | 0.614 | **1.73** | −0.025 | 0.248 |
| Oracle 6-class | 6.42 | 0.618 | 2.27 | 0.041 | 0.287 |

Text-metric gains are tiny (+0.22 chrF, +0.006 BERT-F1) — not robust given n=11 and a single seed. Paradoxically, both oracle models follow the target emotion flow *worse* than base: trajectory correlation falls 0.156 → −0.025 / 0.041, and line-level agreement falls 0.326 → 0.248 / 0.287. Collapsing 44→6 classes does not recover tracking, so label resolution is not the bottleneck.

**Conclusion:** At GPT-2 small scale, text-prefix emotion conditioning — regardless of format, resolution, or amount of information — does not reliably ground emotion control in generation. Loss is applied to both the trajectory block and the poem text, so the model can learn the block without tying its *i*-th value to the *i*-th generated line.

---

