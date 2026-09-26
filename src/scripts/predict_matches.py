"""
Predict matches from the candidate pairs using the trained LightGBM model.

Reads:  output/candidate_pairs_raw.tsv  (internal one-pair-per-row format)
Writes: output/matching_results.tsv     (competition format)

If candidate_pairs_raw.tsv does not exist but candidate_pairs.tsv does in the
old one-pair-per-row format, it falls back to that.
"""
import sys
import time
import pickle
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from business_entity_resolution.data import load_source, TRAIN_DIR, TEST_DIR
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.features import build_feature_matrix


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict entity matches from candidates")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override prediction threshold (default: use model's optimized value)",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    data_dir = TEST_DIR if args.mode == "test" else TRAIN_DIR

    # Resolve candidate file: prefer raw, fall back to old format
    raw_file = OUTPUT_DIR / "candidate_pairs_raw.tsv"
    old_file = OUTPUT_DIR / "candidate_pairs.tsv"

    if raw_file.exists():
        candidates_file = raw_file
    elif old_file.exists():
        candidates_file = old_file
    else:
        print("Error: No candidate file found. Run generate_candidates.py first.")
        return

    out_file = OUTPUT_DIR / "matching_results.tsv"

    # ── Step 1: Load and normalize data ──
    print(f"Step 1: Loading data ({args.mode})...")
    cols = ["entity_id", "business_name", "business_address", "country"]
    t0 = time.time()

    s1 = load_source(data_dir / f"{args.mode}_source1.tsv", usecols=cols)
    s2 = load_source(data_dir / f"{args.mode}_source2.tsv", usecols=cols)
    s3 = load_source(data_dir / f"{args.mode}_source3.tsv", usecols=cols)

    s1 = normalize_dataframe(s1, cols)
    s2 = normalize_dataframe(s2, cols)
    s3 = normalize_dataframe(s3, cols)

    target_df = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    s1_lookup = s1.set_index("entity_id")
    target_lookup = target_df.set_index("entity_id")
    all_s1_ids = s1["entity_id"].values.tolist()
    print(f"  Loaded in {time.time()-t0:.1f}s  (S1={len(s1):,}, targets={len(target_df):,})")

    # ── Step 2: Load model ──
    print("Step 2: Loading LightGBM model...")
    model_path = MODELS_DIR / "lgbm_model.pkl"
    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    if isinstance(saved, dict):
        model = saved["model"]
        optimal_threshold = saved.get("threshold", 0.4)
    else:
        model = saved
        optimal_threshold = 0.4

    threshold = args.threshold if args.threshold is not None else optimal_threshold
    print(f"  Threshold: {threshold}")

    # ── Step 3: Stream candidates, extract features, predict ──
    print(f"Step 3: Streaming {candidates_file.name} and predicting...")
    matches_dict = defaultdict(list)
    chunk_size = 500_000

    chunk_iter = pd.read_csv(candidates_file, sep="\t", chunksize=chunk_size, dtype=str)

    t_start = time.time()
    total_processed = 0
    total_matches = 0

    for i, chunk in enumerate(chunk_iter):
        t_chunk = time.time()

        # Support both column naming conventions
        if "s1_id" in chunk.columns:
            pass  # columns already named s1_id / target_id
        elif "source1_entity_id" in chunk.columns:
            # Competition grouped format — need to explode
            rows = []
            for s1_id, cands_str in zip(chunk["source1_entity_id"], chunk["candidate_entity_ids"]):
                if pd.isna(cands_str) or str(cands_str).strip() == "":
                    continue
                for t_id in str(cands_str).split(","):
                    t_id = t_id.strip()
                    if t_id:
                        rows.append({"s1_id": s1_id, "target_id": t_id})
            chunk = pd.DataFrame(rows)
            if chunk.empty:
                continue
        else:
            # Unknown columns — assume first two are s1 and target
            chunk.columns = ["s1_id", "target_id"]

        # Drop pairs where either ID is missing from lookups
        valid = chunk["s1_id"].isin(s1_lookup.index) & chunk["target_id"].isin(target_lookup.index)
        chunk = chunk[valid].reset_index(drop=True)

        if chunk.empty:
            continue

        # Extract features
        X = build_feature_matrix(chunk, s1_lookup, target_lookup)

        # Predict
        y_prob = model.predict_proba(X)[:, 1]
        is_match = y_prob >= threshold

        # Accumulate
        match_rows = chunk[is_match]
        for s1_id, target_id in zip(match_rows["s1_id"], match_rows["target_id"]):
            matches_dict[s1_id].append(target_id)

        total_processed += len(chunk)
        total_matches += int(is_match.sum())

        elapsed = time.time() - t_chunk
        print(f"  Chunk {i+1}: {len(chunk):,} pairs → {int(is_match.sum()):,} matches ({elapsed:.1f}s)")

    print(f"\n  Prediction done in {time.time()-t_start:.1f}s. "
          f"Processed {total_processed:,} pairs → {total_matches:,} matches.")

    # ── Step 4: Write matching_results.tsv (competition format) ──
    print("Step 4: Writing matching_results.tsv...")

    # Deduplicate match lists
    for s1_id in matches_dict:
        matches_dict[s1_id] = list(dict.fromkeys(matches_dict[s1_id]))

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = matches_dict.get(s1_id, [])
            joined = ",".join(matched) if matched else ""
            f.write(f"{s1_id}\t{joined}\n")

    matched_entities = sum(1 for s in all_s1_ids if matches_dict.get(s))
    print(f"  Written {len(all_s1_ids):,} rows ({matched_entities:,} with matches)")
    print(f"\nDone! Results saved to {out_file}")


if __name__ == "__main__":
    main()
