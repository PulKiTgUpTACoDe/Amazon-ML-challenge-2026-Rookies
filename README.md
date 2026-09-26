# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge (Business Entity Resolution). 

Our pipeline resolves business entities across 3 independent data sources (12M+ records) on consumer hardware (16GB RAM) using a strict 2-stage approach: multi-strategy blocking followed by gradient-boosted classification.

---

## 1. Architecture Overview

Entity Resolution on this scale (2.2M S1 records against 10.3M Target records) yields over **22 Trillion** possible combinations. We solve this with a **2-Stage Pipeline**:

1. **Stage 1: Multi-Strategy Blocking (Candidate Generation).** We combine three complementary blocking strategies to maximize recall while keeping the candidate set small:
   - **TF-IDF Character N-Gram Blocking** on `business_name` (top_k=10, threshold=0.3) — catches fuzzy name matches
   - **TF-IDF Character N-Gram Blocking** on `business_address` (top_k=5, threshold=0.35) — catches same-address, different-name entities
   - **Exact Match Blocking** on normalized `business_name` — catches trivially identical names

2. **Stage 2: Machine Learning Classification.** For each candidate pair, we compute 27+ string-distance features using RapidFuzz (C++ backend) and feed them into a LightGBM classifier with an F0.5-optimized decision threshold.

---

## 2. Codebase Structure & Flow

### `src/business_entity_resolution/` (Core Engine)

| Module | Purpose |
|--------|---------|
| `data.py` | Safe TSV loading with string typing and schema validation |
| `normalization.py` | Aggressive text cleaning: lowercasing, unicode normalization, abbreviation expansion, whitespace collapse |
| `abbreviations.py` | 40+ business and address abbreviation patterns (Corp→Corporation, St→Street, etc.) |
| `blocking.py` | Multi-strategy blocking: TF-IDF, token overlap, phonetic (Soundex), exact match |
| `features.py` | 27 similarity features: Jaro-Winkler, Levenshtein, token set/sort ratio, partial ratio, Jaccard, containment, numeric match, country match, address presence |
| `evaluation.py` | Per-entity macro-average F0.5 computation |

### `src/scripts/` (Pipeline Steps)

| Script | Purpose |
|--------|---------|
| `train_final_model.py` | Trains LightGBM with 5-fold GroupKFold CV + threshold optimization |
| `generate_candidates.py` | Multi-strategy blocking on full test set (streaming, chunked) |
| `predict_matches.py` | Scores candidate pairs with trained model, outputs final matches |
| `evaluate_model.py` | End-to-end evaluation on a training holdout sample |
| `reformat_output.py` | Converts internal pair format to competition submission format |
| `create_submission.py` | Packages the final zip archive |

---

## 3. How to Run

### Prerequisites
- **RAM:** 16 GB minimum
- **Python:** 3.9+
- **OS:** Windows / Linux / macOS

### Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### Data Placement
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### Run Pipeline (Sequential)

```bash
# 1. Train the model (~15-30 min)
python src/scripts/train_final_model.py

# 2. Generate candidate pairs (~30-60 min)
python src/scripts/generate_candidates.py --mode test

# 3. Predict final matches (~15-20 min)
python src/scripts/predict_matches.py --mode test

# 4. Validate submission
python dataset/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 5. Package for submission
python src/scripts/create_submission.py
```

---

## 4. Output Files

| File | Format | Description |
|------|--------|-------------|
| `output/candidate_pairs.tsv` | `source1_entity_id\tcandidate_entity_ids` | Competition format: one row per S1 entity, comma-separated candidate IDs |
| `output/candidate_pairs_raw.tsv` | `s1_id\ttarget_id` | Internal format: one pair per row (used by predict_matches.py) |
| `output/matching_results.tsv` | `source1_entity_id\tmatched_entity_ids` | Final submission: one row per S1 entity, comma-separated matched IDs |

---

## 5. Key Design Decisions

- **Precision over Recall:** F0.5 weights precision 2x over recall. Our threshold is optimized specifically for this metric.
- **Character N-Grams over Word-Level TF-IDF:** Handles typos, abbreviations, and transliterations better.
- **Precomputed S1 Matrices:** S1 TF-IDF matrices are computed once and reused across all target chunks, cutting blocking time in half.
- **Normalization Before Vectorizer Fit:** Ensures the vocabulary matches the normalized text, preventing feature mismatch.
- **LightGBM:** Natively handles NaN/missing values (critical for Source 3 which lacks addresses), fast CPU inference, and excellent calibration for threshold tuning.
