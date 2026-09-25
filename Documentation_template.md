# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Rookies  
**Team Members:** Pulkit  
**Submission Date:** September 2026

---

## 1. Executive Summary
Our solution implements a highly optimized, memory-efficient Machine Learning pipeline that effectively handles corrupted, unstructured business entity records. By leveraging a fast TF-IDF character n-gram blocking strategy and a LightGBM classification model augmented with RapidFuzz string similarity features, we achieved high recall and precise matching within strict 16GB RAM constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
The datasets (S1, S2, and S3) contained millions of records with extreme noise patterns including misspelled names, missing fields (e.g., S3 completely lacks addresses), abbreviation inconsistencies, and formatting differences. Standard exact-matching approaches yielded exceptionally low recall. Furthermore, the massive size of S2 and S3 (over 10 million rows combined) posed a significant risk of out-of-memory errors on standard hardware, necessitating a heavily streamed and chunked architecture.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier  
**Core Innovation:** Memory-safe chunked Sparse Matrix Cosine Similarity for Blocking, paired with ultra-fast RapidFuzz string-distance extraction. By sampling the TF-IDF vocabulary to 500k records and chunking the dense sparse dot-products, we bypassed Scikit-learn memory crashes and executed the full pipeline in minimal time on a CPU.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** TF-IDF Cosine Similarity on `business_name` using Character N-grams (n=2 to 4).
- **Candidate pairs generated:** Varies by target dataset (dynamically filtered).
- **How you ensured true matches were not lost:** Character n-grams are highly resilient to spelling errors, insertions, and deletions compared to exact word-matching. By using a 0.4 cosine similarity threshold on these n-grams across the `top_k=5` candidates, we maximized the probability that slightly corrupted names are always captured as candidates.

---

## 4. Matching Model

**Features used:**
- Name features: Exact Match Flag, Jaro-Winkler, Levenshtein Normalized Ratio, Token Set Ratio, Length Difference.
- Address features: Exact Match Flag, Jaro-Winkler, Levenshtein Normalized Ratio, Token Set Ratio, Length Difference. (Missing addresses in S3 are gracefully padded with `-1`).
- Other: Country exact match flag, Country missing flag.

**Model type:** LightGBM (`LGBMClassifier`)  
**Threshold selection method:** 0.5 prediction probability threshold. During local validation, `GroupShuffleSplit` on `source1_entity_id` was utilized to completely prevent data leakage and rigorously evaluate F_0.5 optimization.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Strong validation performance matching local benchmarks.
- **Common false positives (wrong merges):** Entities sharing identical generic names (e.g., "Starbucks") in missing-address scenarios, causing the model to over-rely on name similarity without spatial differentiation.
- **Common false negatives (missed matches):** Extreme abbreviations that don't share character n-grams (e.g., "IBM" vs "International Business Machines") escaping the blocking phase.

---

## 6. Conclusion
We successfully designed and deployed a robust, end-to-end entity resolution pipeline capable of scaling to tens of millions of records on consumer hardware. The careful orchestration of memory chunking, sparse matrix operations, and gradient-boosted decision trees proved highly effective at tackling severe real-world data corruption.

---

## Appendix

### A. Code Artefacts
Our complete, runnable code ships in the submission zip under `code/business_entity_resolution/`.
- `src/business_entity_resolution/`: Contains the core logic for data loading, normalization, feature extraction, and TF-IDF blocking.
- `scripts/train_final_model.py`: Trains the LightGBM model by sampling ground truth positives and generating hard negatives via blocked S1/Target chunks.
- `scripts/generate_candidates.py`: Safely streams Target data in 1M chunks to produce `output/candidate_pairs.tsv` using chunked dot-products (chunk size 100).
- `scripts/predict_matches.py`: Streams candidates, generates RapidFuzz features, and outputs final predictions to `output/matching_results.tsv`.

### B. Additional Results
*(Any final metrics or charts can be appended here after test-set evaluation)*

---
