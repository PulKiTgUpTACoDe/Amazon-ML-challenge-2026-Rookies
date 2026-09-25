# Business Entity Resolution Pipeline

This repository contains the end-to-end pipeline for the Amazon ML Challenge (Business Entity Resolution).

## Hardware Requirements
- **RAM:** 16 GB minimum
- **Disk:** ~5-10 GB free space (for output pairs and models)
- **OS:** Cross-platform (tested on Windows/Linux)

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Data placement:
   Ensure your datasets are placed in `dataset/train/` and `dataset/test/` respectively.
   - `dataset/train/train_source1.tsv`
   - `dataset/train/train_source2.tsv`
   - `dataset/train/train_source3.tsv`
   - `dataset/train/train_ground_truth.tsv`
   - `dataset/test/test_source1.tsv`
   - `dataset/test/test_source2.tsv`
   - `dataset/test/test_source3.tsv`

## Execution Steps

The pipeline is split into three parts to strictly adhere to memory constraints (16GB RAM) and allow robust streaming of the massive datasets.

### Step 1: Train the Final Model
```bash
python scripts/train_final_model.py
```
*What this does:* Loads training data, performs TF-IDF blocking on a sample to generate hard negatives, samples positive matches from the ground truth, extracts 15+ string similarity features using RapidFuzz, and trains an optimized LightGBM classifier. Saves the model to `models/lgbm_model.pkl`.

### Step 2: Generate Candidate Pairs (Blocking)
```bash
python scripts/generate_candidates.py --mode test
```
*What this does:* Loads the `test` data. Uses a custom chunked Sparse Matrix Dot Product approach to compare all records in Source 1 against Source 2 and Source 3 in chunks. Uses Character-level N-Grams (2-4). Writes the top matching candidates to `output/candidate_pairs.tsv` incrementally. Memory usage peaks at ~2-3 GB.

### Step 3: Predict Matches (Evaluation)
```bash
python scripts/predict_matches.py --mode test --threshold 0.5
```
*What this does:* Streams the generated `candidate_pairs.tsv`, looks up the original strings, calculates the exact RapidFuzz features used in training, applies the LightGBM model, and outputs predictions >= 0.5 to `output/matching_results.tsv` in the correct competition format.

### Step 4: Package Submission (Optional)
```bash
python scripts/create_submission.py
```
*What this does:* Gathers the code, documentation, and the `output/` files and zips them into a compliant submission package.
