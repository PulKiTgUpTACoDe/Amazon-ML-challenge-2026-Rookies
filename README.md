# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge (Business Entity Resolution). 

Our pipeline is specifically engineered to process massive, highly corrupted text datasets (over 12 million business records) directly on consumer hardware by heavily utilizing chunked data streaming and sparse matrix memory management.

---

## 1. Architecture Overview

Entity Resolution on this scale ($2.2 \times 10^6$ S1 records against $10.3 \times 10^6$ Target records) yields over **22 Trillion** possible combinations. Processing this directly is impossible. We solved this using a strict **2-Stage Pipeline**:

1. **Stage 1: Blocking (Candidate Generation).** We use highly optimized TF-IDF Character N-Grams and chunked Sparse Matrix Cosine Similarity to quickly discard 99.99% of combinations, keeping only the `top_k=5` best candidates per entity.
2. **Stage 2: Machine Learning Classification.** For the remaining candidate pairs, we calculate 15+ advanced string distance metrics (using RapidFuzz) and feed them into a Gradient Boosted Decision Tree (LightGBM) to make the final "Match" or "No-Match" decision.

---

## 2. Codebase Structure & Flow

The codebase is split into two halves: the `src/` directory containing the modular engine, and the `scripts/` directory containing the execution flows.

### `src/business_entity_resolution/` (The Core Engine)
- **`data.py`**
  - *Function `load_source()`:* Responsible for reading the massive TSV files safely, ensuring specific string typing to avoid Pandas memory bloat.
  - *Function `parse_matched_ids()`:* Unpacks the pipe-separated ground truth labels safely into Python sets.
- **`normalization.py`**
  - *Function `normalize_dataframe()`:* Aggressively cleans business strings (lowercasing, punctuation stripping, whitespace trimming). *Why?* To perfectly align slight typos before the string similarity algorithms are applied.
- **`blocking.py`**
  - *Function `block_tfidf()`:* The heart of Candidate Generation. Uses Character N-Grams (2-4) to learn text combinations. Transforms datasets into Sparse Matrices, and computes a Cosine Dot Product (`X1.dot(X2.T)`). 
  - *Why chunked?* It chunks the Dot Product in batches of 100 rows. Without chunking, multiplying 5000 strings against 2,000,000 strings produces an 8.5 billion element dense-sparse matrix, instantly crashing a 16GB PC (requires 68GB RAM). Chunking restricts peak RAM to ~1.5GB.
- **`features.py`**
  - *Function `compute_string_features()` & `build_feature_matrix()`:* Leverages the ultra-fast C++ `RapidFuzz` library to calculate features like Jaro-Winkler, Levenshtein, and Token Set ratios. 
  - *Why Token Set Ratio?* It elegantly matches out-of-order words (e.g., "Starbucks Inc" vs "Inc Starbucks" gets a perfect 1.0 score). It also handles missing target addresses (Source 3) by padding them with a `-1` penalty flag.

### `scripts/` (The Execution Flow)
- **`train_final_model.py`**
  - *What it does:* The Model Trainer. It extracts 100,000 positive matches from the Ground Truth, and generates "Hard Negatives" by blocking 30,000 S1 queries against a random 2,000,000 row sample of S2/S3. It then computes the RapidFuzz features and trains a `LightGBM` model.
  - *Why LightGBM?* It is lightning fast, heavily multi-threaded for CPUs, and natively handles NaN/Missing values exceptionally well.
- **`generate_candidates.py`**
  - *What it does:* The Heavy Lifter. It streams the massive 10.3 million row test-set targets in 1,000,000 row chunks, blocking them against S1 in memory, and appending candidate pairs sequentially to `output/candidate_pairs.tsv`.
- **`predict_matches.py`**
  - *What it does:* The Predictor. It streams the `candidate_pairs.tsv` file in 500,000 row chunks, extracts the exact RapidFuzz features dynamically, runs them through the trained LightGBM model, and formats the output into the final submission TSV format.
- **`create_submission.py`**
  - *What it does:* Automatically bundles all code, outputs, and documentation into the strict `teamname_submission.zip` structure required by the competition rules.

---

## 3. How to Setup and Run on Your Local PC

### Step 1: Hardware & Environment Prep
- **RAM:** 16 GB minimum required.
- **Python:** Python 3.9+ recommended.
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
  pip install -r requirements.txt
  ```

### Step 2: Data Placement
Ensure your Amazon ML Challenge dataset files are extracted and placed exactly in this folder structure:
```text
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

### Step 3: Run the Pipeline (Sequentially)

**1. Train the Production Model**
*(This samples the data, extracts features, and trains/saves `models/lgbm_model.pkl`)*
```bash
python scripts/train_final_model.py
```

**2. Generate Candidate Pairs**
*(This streams the test datasets and blocks them against S1, saving to `output/candidate_pairs.tsv`. This is the longest running script.)*
```bash
python scripts/generate_candidates.py --mode test
```

**3. Predict Final Matches**
*(This streams the generated candidates, applies the LightGBM model, and outputs predictions to `output/matching_results.tsv`)*
```bash
python scripts/predict_matches.py --mode test --threshold 0.5
```

**4. Create the Zip Submission**
*(This safely zips up the `output/`, `code/`, `README.md`, `requirements.txt`, and `Documentation_template.md` precisely how the competition rules dictated)*
```bash
python scripts/create_submission.py
```
