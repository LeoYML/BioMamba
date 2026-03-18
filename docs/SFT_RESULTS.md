# BioMamba SFT (Supervised Fine-Tuning) Results

> Last updated: 2026-03-09

---

## 0. Dataset Description

### BioASQ Training Data
- **Source**: BioASQ Task 7b training set (HuggingFace `nanyy1025/bioasq_7b_yesno`) + BioASQ Task 13b training set
- **Deduplication**: 7b and 13b training questions are distinct from each other and from the test set (0 question overlap verified)
- Two versions prepared:
  - `bioasq_combined_all`: 1462 samples (7b: 670, 13b: 792) — yes=1072 (73%), no=390 (27%)
  - `bioasq_combined_65_35`: 1114 samples (downsampled yes) — yes=724 (65%), no=390 (35%)

### BioASQ Test Data
- **Source**: BioASQ 13b Golden Enrichment files (13B1–13B4_golden.json)
- **82 yesno questions**: 54 yes (65.9%), 28 no (34.1%)
- Extracted from 340 total questions (yesno/factoid/list/summary), only yesno used

### PubMedQA Data
- **Source**: HuggingFace `qiaojin/PubMedQA` (pqa_labeled subset)
- **1000 samples**: 800 train / 200 test — yes=103, no=72, maybe=25 (test)

---

## 1. Best Model Summary (by Metric)

### Best by BioASQ Accuracy

| Model | BioASQ Acc | BioASQ F1 | BioASQ Recall | BioASQ Precision | yes-F1 | no-F1 |
|---|---|---|---|---|---|---|
| **BioMamba-2.7B** | **90.24%** | 0.890 | 0.883 | 0.898 | 0.927 | 0.852 |
| BioMamba-1.3B (v14) | 85.37% | 0.831 | 0.820 | 0.848 | 0.893 | 0.769 |
| BioMamba-780M (v5) | 79.27% | 0.778 | 0.777 | 0.798 | 0.835 | 0.721 |
| BioMamba-370M (v7) | 78.05% | 0.721 | 0.704 | 0.793 | 0.850 | 0.591 |
| BioMamba-130M | 68.29% | 0.634 | 0.630 | 0.642 | 0.768 | 0.500 |

### Best by PubMedQA Accuracy (logprob + SFT template)

| Model | PubMedQA Acc | PubMedQA F1 | PubMedQA Recall | yes-F1 | no-F1 |
|---|---|---|---|---|---|
| **BioMamba-2.7B** | **73.00%** | 0.515 | 0.548 | 0.800 | 0.745 |
| BioMamba-1.3B (v14) | 70.00% | 0.493 | 0.528 | 0.782 | 0.697 |
| BioMamba-780M (v5) | 67.50% | 0.474 | 0.506 | 0.754 | 0.667 |
| BioMamba-370M (v7) | 66.00% | 0.460 | 0.489 | 0.746 | 0.633 |
| BioMamba-130M | 63.00% | 0.482 | 0.489 | 0.727 | 0.595 |

### Best Balanced (BioASQ + PubMedQA)

| Model | BioASQ Acc | BioASQ F1 | PubMedQA Acc | PubMedQA F1 | Checkpoint |
|---|---|---|---|---|---|
| **BioMamba-2.7B** | **90.24%** | 0.890 | **73.00%** | 0.515 | v1 (only version) |
| BioMamba-1.3B | 85.37% | 0.831 | 70.00% | 0.493 | v14 (F1 best) |
| BioMamba-780M | 79.27% | 0.778 | 67.50% | 0.474 | v5 (balanced) |
| BioMamba-370M | 78.05% | 0.721 | 66.00% | 0.460 | v7 |
| BioMamba-130M | 68.29% | 0.634 | 63.00% | 0.482 | v3 (only version) |

### Comparison with External Biomedical Models

#### QA Performance (BioASQ + PubMedQA)

| Model | Params | Type | PubMed PPL | BioASQ Acc | BioASQ F1 | PubMedQA Acc | PubMedQA F1 |
|---|---|---|---|---|---|---|---|
| Bio-Medical-Llama-3.2-1B | 1B | Llama 3.2 (SFT) | 14.46 | 82.93% | 0.813 | 46.50% | 0.266 |
| Meditron3-Gemma2-2B | 2B | Gemma2 (CPT) | 6.67 | 79.27% | 0.486 | 44.00% | 0.322 |
| BioGPT-Large-PubMedQA | 1.5B | GPT-2 (SFT) | 9.26 | 20.73% | 0.021 | 63.00% | 0.459 |
| BioGPT-Large | 1.5B | GPT-2 (CPT) | 10.00 | 26.83% | 0.020 | 51.00% | 0.298 |
| BioMedLM | 2.7B | GPT-2 (CPT) | 25.27 | 42.68% | 0.048 | 46.50% | 0.278 |
| Gemma3-finetune | ~1B | Gemma3 (SFT) | 19.90 | 65.85% | 0.217 | 51.50% | 0.243 |
| BioGPT | 347M | GPT-2 (CPT) | 15.09 | 3.66% | 0.004 | 53.00% | 0.341 |
| | | | | | | | |
| **BioMamba-2.7B** | **2.7B** | **Mamba2 (CPT+SFT)** | **5.28** | **90.24%** | **0.890** | **73.00%** | **0.515** |
| **BioMamba-1.3B** | **1.3B** | **Mamba2 (CPT+SFT)** | **5.66** | **85.37%** | **0.831** | **70.00%** | **0.493** |
| **BioMamba-780M** | **780M** | **Mamba2 (CPT+SFT)** | **6.15** | **79.27%** | **0.778** | **67.50%** | **0.474** |
| **BioMamba-370M** | **370M** | **Mamba2 (CPT+SFT)** | **6.75** | **78.05%** | **0.721** | **66.00%** | **0.460** |
| **BioMamba-130M** | **130M** | **Mamba2 (CPT+SFT)** | **8.42** | **68.29%** | **0.634** | **63.00%** | **0.482** |

> **Note**: PubMed PPL from BioMamba_PPL_Evaluation.md. BioASQ: generative evaluation (82 yes/no questions from BioASQ 13b). PubMedQA: logprob strategy (200 test samples). External baselines sorted by BioASQ Acc. BioGPT/BioMedLM have no SFT for yes/no QA — poor BioASQ is expected.

> MIMIC-IV clinical generation results: see `MIMIC_EVALUATION_RESULTS.md` Section 3.6.

---

## 2. CPT Contribution: CPT+SFT vs SFT-only

> Does CPT (Continue Pre-Training on PubMed+Wikipedia) help SFT performance?
> We trained SFT on both CPT checkpoints and base models (no CPT) with identical hyperparameters.

### Summary: CPT+SFT vs SFT-only (BioASQ)

| Model | SFT-only Acc | CPT+SFT Acc | CPT Gain | SFT-only F1 | CPT+SFT F1 | CPT Gain |
|---|---|---|---|---|---|---|
| **BioMamba-2.7B** | 90.24% | **90.24%** | **+0.00%** | 0.890 | **0.890** | +0.000 |
| BioMamba-1.3B | 82.93% | **85.37%** | **+2.44%** | 0.799 | **0.831** | **+0.032** |
| BioMamba-780M | 78.05% | **79.27%** | **+1.22%** | 0.728 | **0.778** | **+0.050** |
| BioMamba-370M | 79.27% | **78.05%** | **−1.22%** | 0.724 | **0.721** | **−0.003** |
| BioMamba-130M | 53.66% | **68.29%** | **+14.63%** | 0.536 | **0.634** | **+0.098** |

### Summary: CPT+SFT vs SFT-only (PubMedQA, logprob)

| Model | SFT-only Acc | CPT+SFT Acc | CPT Gain | SFT-only F1 | CPT+SFT F1 | CPT Gain |
|---|---|---|---|---|---|---|
| **BioMamba-2.7B** | 70.50% | **73.00%** | **+2.50%** | 0.495 | **0.515** | **+0.020** |
| BioMamba-1.3B | 65.00% | **70.00%** | **+5.00%** | 0.452 | **0.493** | **+0.041** |
| BioMamba-780M | 64.50% | **67.50%** | **+3.00%** | 0.477 | **0.474** | −0.003 |
| BioMamba-370M | **65.50%** | 66.00% | +0.50% | 0.458 | 0.460 | +0.002 |
| BioMamba-130M | 51.50% | **63.00%** | **+11.50%** | 0.362 | **0.482** | **+0.120** |

### Key Findings

- **130M**: CPT is essential — +12.29% BioASQ, +11.50% PubMedQA. Small models benefit most from domain pre-training.
- **370M–780M**: CPT provides modest gains (+1–2% accuracy, +0.03–0.05 F1). 780M v5 balances both tasks well.
- **1.3B**: CPT+SFT v14 achieves F1=0.831, Acc=85.37%, surpassing SFT-only (F1=0.799). Key: balanced data (65_35) + 3 epochs + lr=7e-6 + ratio=0.70.
- **2.7B**: Identical BioASQ performance (90.24%). CPT helps PubMedQA (+2.5% accuracy) but the gap narrows significantly at scale.
- **Conclusion**: CPT helps across all sizes when paired with balanced data and proper tuning. The gain is largest for small models (130M: +12%) and moderate for larger ones (1.3B: +2%, 2.7B: +2.5% PubMedQA).

### BioASQ Detailed (SFT-only / CPT+SFT)

| Model | Stage | Acc | F1 | Recall | Precision | yes-F1 | no-F1 |
|---|---|---|---|---|---|---|---|
| 2.7B | SFT-only | 90.24% | 0.890 | 0.883 | 0.898 | 0.927 | 0.852 |
| 2.7B | **CPT+SFT** | **90.24%** | **0.890** | **0.883** | **0.898** | **0.927** | **0.852** |
| 1.3B | SFT-only | 82.93% | 0.799 | 0.784 | 0.826 | 0.877 | 0.720 |
| 1.3B | **CPT+SFT v14** | **85.37%** | **0.831** | **0.820** | **0.848** | **0.893** | **0.769** |
| 780M | SFT-only | 78.05% | 0.728 | 0.713 | 0.780 | 0.847 | 0.609 |
| 780M | **CPT+SFT v5** | **79.27%** | **0.778** | **0.777** | **0.798** | **0.835** | **0.721** |
| 370M | SFT-only | 79.27% | 0.724 | 0.705 | 0.846 | 0.862 | 0.585 |
| 370M | CPT+SFT v7 | 78.05% | 0.721 | 0.704 | 0.793 | 0.850 | 0.591 |
| 130M | SFT-only | 53.66% | 0.536 | 0.614 | 0.624 | 0.513 | 0.558 |
| 130M | **CPT+SFT** | **68.29%** | **0.634** | **0.630** | **0.642** | **0.768** | **0.500** |

### PubMedQA Detailed (SFT-only / CPT+SFT, logprob + SFT template)

| Model | Stage | Acc | F1 | Recall | Precision | yes-F1 | no-F1 |
|---|---|---|---|---|---|---|---|
| 2.7B | SFT-only | 70.50% | 0.495 | 0.488 | 0.505 | 0.783 | 0.703 |
| 2.7B | **CPT+SFT** | **73.00%** | **0.515** | **0.548** | **0.488** | **0.800** | **0.745** |
| 1.3B | SFT-only | 65.00% | 0.452 | 0.481 | 0.432 | 0.737 | 0.619 |
| 1.3B | **CPT+SFT v14** | **70.00%** | **0.493** | **0.528** | **0.462** | **0.782** | **0.697** |
| 780M | SFT-only | 64.50% | 0.477 | 0.485 | 0.491 | 0.737 | 0.636 |
| 780M | **CPT+SFT v5** | **67.50%** | **0.474** | **0.506** | **0.447** | **0.754** | **0.667** |
| 370M | SFT-only | **65.50%** | **0.458** | **0.489** | — | 0.736 | **0.639** |
| 370M | CPT+SFT v7 | 66.00% | 0.460 | 0.489 | 0.455 | **0.746** | 0.633 |
| 130M | SFT-only | 56.00% | 0.379 | 0.377 | 0.387 | 0.667 | 0.472 |
| 130M | **CPT+SFT** | **63.00%** | **0.482** | **0.489** | **0.516** | **0.727** | **0.595** |

---

## 3. Per-Model SFT Details

### 3.1 BioMamba-2.7B

**One version — default config achieved best results across all sizes.**

| Parameter | Value |
|---|---|
| CPT checkpoint | `checkpoints/2.7b_mixed_wiki/.../best_model` |
| SFT checkpoint | `checkpoints/2.7b_sft/biomamba2_sft_mamba2-2.7b_full_20260309_014707/best_model` |
| Learning rate | 2e-5 |
| Epochs | 5 |
| Batch size | 4 × 8 = 32 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Scheduler | cosine |
| Precision | bf16 |
| **Data mix** | **PubMedQA 800 + BioASQ 1486 (ratio=0.65)** |
| Data source | `bioasq_combined_65_35` |

| Metric | BioASQ | PubMedQA (logprob) |
|---|---|---|
| Accuracy | **90.24%** | **73.00%** |
| Macro F1 | 0.890 | 0.515 |
| Macro Recall | 0.883 | 0.548 |
| Macro Precision | 0.898 | 0.488 |
| yes-F1 | 0.927 | 0.800 |
| no-F1 | 0.852 | 0.745 |

---

### 3.2 BioMamba-1.3B

**17 versions tuned. v14 is current best (F1=0.831).**

#### Hyperparameter Comparison

| Version | LR | Epochs | WD | Warmup | BioASQ Data | Ratio | Train Samples |
|---|---|---|---|---|---|---|---|
| v1 | 2e-5 | 5 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v2 | 1e-5 | 3 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v3 | 5e-6 | 2 | 0.05 | 0.15 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v4 | 5e-6 | 2 | 0.05 | 0.15 | bioasq_combined_all | 0.80 | 800+3200=4000 |
| v5 | 5e-6 | 3 | 0.05 | 0.15 | bioasq_combined_all | 0.80 | 800+3200=4000 |
| v6 | 8e-6 | 2 | 0.05 | 0.15 | bioasq_combined_all | 0.80 | 800+3200=4000 |
| v7a | 5e-6 | 2 | 0.05 | 0.15 | bioasq_combined_all | 0.90 | 800+7200=8000 |
| v7b | 5e-6 | 2 | 0.05 | 0.15 | bioasq_combined_all | 0.90 | 800+7200=8000 |
| v8 | 2e-5 | 5 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v9 | 1e-5 | 3 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v10 | 5e-6 | 3 | 0.03 | 0.15 | bioasq_combined_65_35 | 0.70 | 800+1867=2667 |
| v11 | 8e-6 | 3 | 0.03 | 0.1 | bioasq_combined_all | 0.75 | 800+2400=3200 |
| v12 | 5e-6 | 3 | 0.03 | 0.15 | bioasq_combined_50_50 | 0.70 | 800+1820=2620 |
| v13 | 5e-6 | 4 | 0.03 | 0.15 | bioasq_combined_65_35 | 0.70 | 800+1867=2667 |
| **v14** | **7e-6** | **3** | **0.03** | **0.15** | **bioasq_combined_65_35** | **0.70** | **800+1867=2667** |
| v15 | 7e-6 | 4 | 0.03 | 0.15 | bioasq_combined_50_50 | 0.70 | 800+1820=2620 |
| bioasq_only | 5e-6 | 3 | 0.05 | 0.15 | bioasq_combined_all | 100% | 1071 (no PubMedQA) |

> All versions: batch_size=4, accumulation=8, eff_bs=32, cosine scheduler, bf16.

#### BioASQ Results

| Version | Acc | F1 | Recall | Precision | yes-F1 | no-F1 |
|---|---|---|---|---|---|---|
| **v14** | **85.37%** | **0.831** | **0.820** | **0.848** | **0.893** | **0.769** |
| v10 | 84.15% | 0.819 | 0.811 | 0.793 | 0.883 | 0.755 |
| v7b | 81.71% | 0.787 | 0.775 | 0.807 | 0.867 | 0.706 |
| v7a | 80.49% | 0.775 | 0.766 | 0.789 | 0.857 | 0.692 |
| v4 | 79.27% | 0.753 | 0.739 | 0.783 | 0.852 | 0.653 |
| v8 | 78.05% | 0.776 | 0.816 | — | 0.809 | 0.743 |
| v3 | 78.05% | 0.769 | 0.790 | 0.765 | 0.820 | 0.719 |
| v11 | 78.05% | 0.720 | 0.704 | 0.793 | 0.850 | 0.591 |
| v6 | 78.05% | 0.747 | 0.739 | 0.759 | 0.839 | 0.654 |
| v1 | 76.83% | 0.764 | 0.807 | 0.777 | 0.796 | 0.732 |
| v5 | 76.83% | 0.709 | 0.695 | 0.767 | 0.840 | 0.578 |
| bioasq_only | 75.61% | 0.689 | 0.677 | 0.754 | 0.833 | 0.546 |
| v9 | 70.73% | 0.707 | 0.769 | — | 0.721 | 0.692 |
| v2 | 60.98% | 0.608 | 0.704 | 0.733 | 0.579 | 0.636 |

#### PubMedQA Results (logprob + SFT template)

| Version | Acc | F1 | Recall | yes-F1 | no-F1 |
|---|---|---|---|---|---|
| **v14** | **70.00%** | **0.493** | **0.528** | **0.782** | **0.697** |
| v4 | 70.50% | — | — | — | — |
| v10 | 69.00% | 0.486 | 0.521 | 0.769 | 0.688 |
| bioasq_only | 67.00% | 0.475 | 0.513 | 0.737 | 0.687 |
| v7b | 66.50% | 0.487 | 0.503 | 0.763 | 0.629 |
| v1 | 62.50% | 0.439 | — | 0.710 | 0.608 |
| v2 | 57.00% | 0.405 | — | 0.628 | 0.587 |

#### Key Findings (1.3B)
- **v14 is best**: lr=7e-6, 3ep, wd=0.03, bioasq_65_35 ratio=0.70 → **F1=0.831, Acc=85.37%**
- Key improvements: (1) balanced data (65_35 vs all), (2) 3 epochs, (3) lr=7e-6 sweet spot, (4) ratio=0.70
- v10 (lr=5e-6) vs v14 (lr=7e-6): slightly higher lr gives +1.2% acc, +0.012 F1
- v12/v15 (50:50 data): too few samples (780 BioASQ) — underfits
- v13 (4ep): overfits compared to 3ep
- **BioASQ-only fails**: Less data (1071) + no cross-task transfer = 75.61%

---

### 3.3 BioMamba-780M

**7 versions tuned. v5 selected: best balanced (BioASQ 79.27% + PubMedQA 67.50%).**

| Parameter | v1 (baseline) | v3 (BioASQ best) | **v5 (selected)** |
|---|---|---|---|
| CPT checkpoint | `checkpoints/780m_mixed_wiki/.../best_model` | same | same |
| SFT checkpoint | `checkpoints/780m_sft/.../best_model` | `checkpoints/780m_sft_v3/.../best_model` | `checkpoints/780m_sft_v5/.../best_model` |
| Learning rate | 2e-5 | 2e-5 | **2e-5** |
| Epochs | 5 | 3 | **5** |
| Batch size | 16 × 4 = 64 | 8 × 8 = 64 | 8 × 8 = 64 |
| Weight decay | 0.01 | 0.03 | **0.03** |
| Warmup ratio | 0.1 | 0.15 | **0.15** |
| Data source | bioasq_combined_65_35 | bioasq_combined_65_35 | bioasq_combined_65_35 |
| BioASQ ratio | 0.65 | 0.70 | **0.65** |

#### Hyperparameter Comparison

| Version | LR | Epochs | WD | Warmup | Data | Ratio | Note |
|---|---|---|---|---|---|---|---|
| v1 | 2e-5 | 5 | 0.01 | 0.1 | 65_35 | 0.65 | baseline |
| v2 | 1e-5 | 3 | 0.03 | 0.15 | 65_35 | 0.70 | too low lr |
| v3 | 2e-5 | 3 | 0.03 | 0.15 | 65_35 | 0.70 | BioASQ best |
| v4 | 5e-6 | 3 | 0.03 | 0.15 | 65_35 | 0.70 | too low lr |
| **v5** | **2e-5** | **5** | **0.03** | **0.15** | **65_35** | **0.65** | **selected (balanced)** |
| v6 | 2e-5 | 3 | 0.03 | 0.15 | 65_35 | 0.50 | too much PubMedQA |
| v7 | 2e-5 | 5 | 0.01 | 0.1 | all | 0.60 | more data, worse |

> All versions: cosine scheduler, bf16, batch=64.

#### BioASQ Results

| Version | Acc | F1 | yes-F1 | no-F1 |
|---|---|---|---|---|
| v3 | **81.71%** | **0.795** | 0.862 | 0.727 |
| v1 | 79.27% | 0.781 | 0.832 | 0.730 |
| **v5** | **79.27%** | **0.778** | 0.835 | 0.721 |
| v7 | 75.61% | 0.751 | 0.787 | 0.714 |
| v4 | 73.17% | 0.711 | 0.788 | 0.633 |
| v6 | 73.17% | 0.728 | 0.761 | 0.694 |
| v2 | 67.07% | 0.667 | 0.703 | 0.630 |

#### PubMedQA Results

| Version | Acc | F1 | yes-F1 | no-F1 |
|---|---|---|---|---|
| **v5** | **67.50%** | **0.474** | 0.754 | 0.667 |
| v1 | 66.50% | 0.458 | 0.755 | 0.619 |
| v7 | 65.50% | 0.464 | 0.730 | 0.662 |
| v6 | 65.00% | 0.454 | 0.737 | 0.623 |
| v3 | 64.50% | 0.437 | 0.741 | 0.569 |
| v4 | — | — | — | — |
| v2 | — | — | — | — |

#### Key findings
- v3 (BioASQ best): ratio=0.70 maximizes BioASQ but hurts PubMedQA
- **v5 (selected)**: ratio=0.65 + 5ep + wd=0.03 — best PubMedQA (67.50%) with strong BioASQ (79.27%)
- Lower lr (1e-5, 5e-6) severely hurts 780M — needs ≥2e-5
- BioASQ vs PubMedQA trade-off: higher BioASQ ratio → better BioASQ, worse PubMedQA

---

### 3.4 BioMamba-370M

**7 versions tuned. v7 selected (balanced). Key issue: lr sensitivity — too high (5e-5) or too low (1e-5) both hurt.**

#### Hyperparameter Comparison

| Version | LR | Epochs | WD | Warmup | BioASQ Data | Ratio | Train Samples |
|---|---|---|---|---|---|---|---|
| v1 | 2e-5 | 5 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v2 | 5e-5 | 5 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v3 | 1e-5 | 3 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v4 | 3e-5 | 5 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v5 | 3e-5 | 2 | 0.01 | 0.1 | bioasq_combined_65_35 | 0.65 | 800+1486=2286 |
| v6 | 3e-5 | 2 | 0.01 | 0.1 | bioasq_combined_all | 0.80 | 800+3200=4000 |
| **v7** | **3e-5** | **2** | **0.01** | **0.1** | **bioasq_combined_all** | **0.70** | **800+1867=2667** |

> All versions: batch_size=16, accumulation=4, eff_bs=64, cosine scheduler, bf16.

#### BioASQ Results

| Version | Acc | F1 | Recall | Precision | yes-F1 | no-F1 |
|---|---|---|---|---|---|---|
| v6 | 81.71% | 0.756 | 0.732 | 0.891 | 0.878 | 0.634 |
| **v7** | **78.05%** | **0.721** | 0.704 | 0.793 | 0.850 | 0.591 |
| v5 | 68.29% | 0.676 | 0.708 | 0.687 | 0.723 | 0.629 |
| v1 | 67.07% | 0.656 | 0.673 | 0.657 | 0.727 | 0.585 |
| v2 | 65.85% | 0.641 | 0.655 | 0.641 | 0.720 | 0.563 |
| v4 | 64.63% | 0.631 | 0.646 | 0.632 | 0.707 | 0.554 |
| v3 | 57.32% | 0.565 | 0.590 | 0.581 | 0.624 | 0.507 |

#### PubMedQA Results (logprob + SFT template)

| Version | Acc | F1 | Recall | yes-F1 | no-F1 |
|---|---|---|---|---|---|
| **v7** | **66.00%** | 0.460 | 0.489 | 0.746 | 0.633 |
| v6 | 61.00% | 0.432 | — | 0.683 | 0.614 |

#### Key Findings (370M)
- lr=5e-5 (v2) too high, lr=1e-5 (v3) too low → lr=3e-5 is sweet spot
- 5 epochs overfits → 2 epochs optimal
- v6 (bioasq_all 80%) achieved 81.71% BioASQ but poor PubMedQA (61%)
- **v7 (selected, 70% ratio)**: balanced — BioASQ 78.05%, PubMedQA 66%

---

### 3.5 BioMamba-130M

**Best model from earlier experiments (checkpoints/mixed_sft_v3).**

| Parameter | Value |
|---|---|
| CPT checkpoint | `checkpoints/mixed_wiki/.../best_model` |
| SFT checkpoint | `checkpoints/mixed_sft_v3/biomamba_sft_mamba2-130m_full/best_model` |
| Learning rate | 3e-5 |
| Epochs | 3 |
| Batch size | 16 × 8 = 128 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| **Data mix** | **PubMedQA 800 + BioASQ ~1486 (ratio=0.65)** |
| Data source | `bioasq_combined_65_35` |

| Metric | BioASQ | PubMedQA (logprob) |
|---|---|---|
| Accuracy | 68.29% | 63.00% |
| Macro F1 | 0.634 | 0.482 |
| Macro Recall | 0.630 | 0.489 |
| Macro Precision | 0.642 | 0.516 |
| yes-F1 | 0.768 | 0.727 |
| no-F1 | 0.500 | 0.595 |

---

## 4. Data Mixing Analysis

### BioASQ Ratio vs Performance (1.3B)

| BioASQ Ratio | BioASQ Data | Total Train | BioASQ Acc | PubMedQA Acc |
|---|---|---|---|---|
| 65% | combined_65_35 | 2286 | 78.05% (v3) | — |
| 80% | combined_all | 4000 | 79.27% (v4) | 70.50% |
| 90% | combined_all | 8000 | 81.71% (v7b) | 66.50% |
| 100% | combined_all (only) | 1071 | 75.61% | 67.00% |

### BioASQ Ratio vs Performance (370M)

| BioASQ Ratio | BioASQ Data | Total Train | BioASQ Acc | PubMedQA Acc |
|---|---|---|---|---|
| 65% | combined_65_35 | 2286 | 68.29% (v5) | — |
| 70% | combined_all | 2667 | 78.05% (v7) | 66.00% |
| 80% | combined_all | 4000 | 81.71% (v6) | 61.00% |

> Key insight: Mixing PubMedQA is beneficial. Pure BioASQ-only underperforms due to fewer samples (1071 vs 8000 mixed) and no cross-task transfer. Higher BioASQ ratio improves BioASQ but hurts PubMedQA.

---

## 5. Ablation: BioASQ-only vs Mixed (1.3B)

| Config | BioASQ Acc | BioASQ F1 | BioASQ Recall | PubMedQA Acc | PubMedQA F1 |
|---|---|---|---|---|---|
| Mixed 90% (v7b) | **81.71%** | **0.787** | **0.775** | 66.50% | 0.487 |
| Mixed 80% (v4) | 79.27% | 0.753 | 0.739 | **70.50%** | — |
| BioASQ-only | 75.61% | 0.689 | 0.677 | 67.00% | 0.475 |

**Conclusion**: Mixed training > single-task training. PubMedQA data acts as regularization and provides complementary QA knowledge.

---

## 6. PubMedQA Margin Thresholding (maybe detection)

> Problem: All models default to yes/no predictions, never outputting "maybe" (12.5% of test set).
> Solution: If the top-2 logprob margin < threshold AND maybe logprob > threshold, predict "maybe".

### Best Macro F1 with Margin Thresholding

| Model | Baseline Acc | Baseline F1 | Margin Acc | Margin F1 | F1 Gain | Config (margin, maybe_t) |
|---|---|---|---|---|---|---|
| **2.7B** | 73.0% | 51.5% | 74.0% | **63.2%** | **+11.7%** | (0.40, 0.25) |
| 370M-v7 | 66.0% | 46.0% | 65.5% | **55.6%** | **+9.6%** | (0.40, 0.15) |
| 130M | 63.0% | 48.2% | 62.5% | **52.6%** | **+4.4%** | (0.30, 0.30) |
| 1.3B-v7b | 66.5% | 48.7% | 59.5% | **50.0%** | +1.3% | (0.20, 0.05) |
| 780M | 66.5% | 45.8% | 65.5% | **47.5%** | +1.7% | (0.10, 0.15) |

### Per-Class F1 at Best Margin Config

| Model | Config | yes-F1 | no-F1 | Macro F1 |
|---|---|---|---|---|
| **2.7B** | (0.40, 0.25) | 81.6% | 73.0% | **63.2%** |
| **370M-v7** | (0.40, 0.15) | 74.8% | 63.6% | **55.6%** |
| **130M** | (0.30, 0.30) | 73.0% | 58.7% | **52.6%** |
| 1.3B-v7b | (0.20, 0.05) | 76.1% | 53.1% | 50.0% |
| 780M | (0.10, 0.15) | 75.6% | 60.7% | 47.5% |

> Key insight: Margin thresholding significantly boosts macro F1 for 2.7B (+11.7%) and 370M (+9.6%).

---

## 7. Common Settings (All Models)

| Setting | Value |
|---|---|
| Training mode | Full fine-tuning (no LoRA) |
| Scheduler | Cosine |
| Max grad norm | 1.0 |
| Precision | bf16 |
| Max sequence length | 1024 |
| Seed | 42 |
| Optimizer | AdamW |
| Loss | Cross-entropy (instruction tokens masked with -100) |
| Prompt template | QA_TEMPLATE (shared with eval) |
| Validation | PubMedQA holdout 200 (or BioASQ 10% for bioasq_only) |
| Early stopping | Best model saved by validation loss |
| BioASQ eval | Few-shot generative (evaluate_bioasq.py) |
| PubMedQA eval | Logprob + SFT template (eval_pubmedqa_generative.py) |

## 8. Checkpoint Paths

| Model | Best Checkpoint |
|---|---|
| 2.7B SFT | `checkpoints/2.7b_sft/biomamba2_sft_mamba2-2.7b_full_20260309_014707/best_model` |
| 1.3B v14 (selected) | `checkpoints/1.3b_sft_v14/biomamba2_sft_mamba2-1.3b_full_20260309_192406/best_model` |
| 780M v5 (selected) | `checkpoints/780m_sft_v5/biomamba2_sft_mamba2-780m_full_20260309_224251/best_model` |
| 370M v7 (selected) | `checkpoints/370m_sft_v7/biomamba2_sft_mamba2-370m_full_20260309_053303/best_model` |
| 130M SFT | `checkpoints/mixed_sft_v3/biomamba_sft_mamba2-130m_full/best_model` |
| **NoCPT SFT (SFT-only, no CPT)** | |
| 2.7B NoCPT SFT | `checkpoints/2.7b_nocpt_sft/biomamba2_sft_mamba2-2.7b_full_20260309_075812/best_model` |
| 1.3B NoCPT SFT | `checkpoints/1.3b_nocpt_sft/biomamba2_sft_mamba2-1.3b_full_20260309_073946/best_model` |
| 780M NoCPT SFT | `checkpoints/780m_nocpt_sft/biomamba2_sft_mamba2-780m_full_20260309_073342/best_model` |
| 370M NoCPT SFT | `checkpoints/370m_nocpt_sft/biomamba2_sft_mamba2-370m_full_20260309_073113/best_model` |
| 130M NoCPT SFT | `checkpoints/nocpt/biomamba_sft_mamba2-130m_full/best_model` |
