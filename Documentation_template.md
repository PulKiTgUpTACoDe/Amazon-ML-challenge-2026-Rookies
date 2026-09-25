# Model Documentation

## 1. Feature Engineering
Our feature engineering relies primarily on advanced string similarity metrics between `business_name`, `business_address`, and `country` pairs. 
- **RapidFuzz Module:** We extract 5 features per column (Jaro-Winkler, Levenshtein Normalized, Partial Ratio, Token Set Ratio, Token Sort Ratio).
- **Missing Value Handling:** Missing target features (e.g., from Source 3 lacking an address) are handled gracefully and padded with `-1`.
- **Normalization:** Before feature extraction, all text is lowercased, stripped of punctuation, whitespace-normalized, and common business suffixes (inc, llc, ltd) are optionally stripped.

## 2. Blocking Strategy
- **TF-IDF Blocking:** We use a `TfidfVectorizer` at the character N-gram level (n=2 to 4).
- **Similarity Search:** A sparse matrix dot-product is calculated between Source 1 and target sources. We enforce a `similarity_threshold` (0.4) and keep the `top_k=5` matches.
- **Chunking:** Because the S2/S3 tables possess upwards of 5 million rows each, we chunk the target TF-IDF transform and dot product operation in segments of 1 million rows, allowing the blocker to run comfortably inside 16GB of RAM while keeping high recall.

## 3. Modeling
- **Algorithm:** LightGBM (`LGBMClassifier`).
- **Data Splitting:** To avoid data leakage, `GroupShuffleSplit` on `source1_entity_id` is used during local validation. 
- **Training Strategy:** The final model is trained on a randomly sampled class-balanced set combining 100,000 true matches from ground truth and hard negatives derived from running the blocking strategy.
- **Hyperparameters:** `learning_rate=0.1`, `max_depth=7`, `n_estimators=100`, `class_weight='balanced'`.

## 4. Hardware Constraints Addressed
- **16GB RAM limit:** Streaming pipelines (`chunksize=1,000,000`) for both candidate generation and final predictions.
- **4GB VRAM limit:** Entirely CPU-based utilizing multi-core LightGBM and highly optimized `scipy.sparse` matrix multiplications.
