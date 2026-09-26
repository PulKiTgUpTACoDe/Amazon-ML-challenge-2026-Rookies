"""
Quick-fix: converts the existing candidate_pairs.tsv from the internal
one-pair-per-row format to the competition-required grouped format.

Before:  s1_id\\ttarget_id          (one pair per row)
After:   source1_entity_id\\tcandidate_entity_ids  (comma-separated, one row per S1 entity)

Also copies the original to candidate_pairs_raw.tsv so predict_matches.py
can still read the pair-level file.
"""
import sys
import time
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from business_entity_resolution.data import load_source, TEST_DIR, TRAIN_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = PROJECT_ROOT / "output"
    data_dir = TEST_DIR if args.mode == "test" else TRAIN_DIR

    current_file = OUTPUT_DIR / "candidate_pairs.tsv"
    raw_backup = OUTPUT_DIR / "candidate_pairs_raw.tsv"

    if not current_file.exists():
        print(f"Error: {current_file} not found!")
        return

    # ── Step 1: Copy existing file to raw backup ──
    print(f"Step 1: Backing up -> {raw_backup.name}")
    shutil.copy2(current_file, raw_backup)

    # ── Step 2: Load all S1 IDs (every one must appear in output) ──
    print(f"Step 2: Loading S1 entity IDs ({args.mode})...")
    s1 = load_source(data_dir / f"{args.mode}_source1.tsv", usecols=["entity_id"])
    all_s1_ids = s1["entity_id"].values.tolist()
    print(f"  {len(all_s1_ids):,} S1 entities")

    # Also load valid S2/S3 IDs for validation
    s2 = load_source(data_dir / f"{args.mode}_source2.tsv", usecols=["entity_id"])
    s3 = load_source(data_dir / f"{args.mode}_source3.tsv", usecols=["entity_id"])
    valid_target_ids = set(s2["entity_id"]) | set(s3["entity_id"])
    del s2, s3
    print(f"  {len(valid_target_ids):,} valid target IDs")

    # ── Step 3: Stream raw file, group and deduplicate ──
    print("Step 3: Grouping pairs by S1 entity (streaming)...")
    candidates = defaultdict(set)
    skipped = 0

    t0 = time.time()
    chunk_iter = pd.read_csv(raw_backup, sep="\t", chunksize=5_000_000, dtype=str)

    for i, chunk in enumerate(chunk_iter):
        col_s1 = chunk.columns[0]
        col_t = chunk.columns[1]

        for s1_id, t_id in zip(chunk[col_s1], chunk[col_t]):
            if pd.notna(s1_id) and pd.notna(t_id):
                t_id = str(t_id).strip()
                if t_id in valid_target_ids:
                    candidates[s1_id].add(t_id)
                else:
                    skipped += 1

        print(f"  Chunk {i+1}: {sum(len(v) for v in candidates.values()):,} unique pairs so far")

    print(f"  Done in {time.time()-t0:.1f}s. Skipped {skipped:,} invalid target IDs.")

    # ── Step 4: Write competition format ──
    print("Step 4: Writing competition-format candidate_pairs.tsv...")
    with open(current_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, set())
            joined = ",".join(sorted(cands)) if cands else ""
            f.write(f"{s1_id}\t{joined}\n")

    unique_pairs = sum(len(v) for v in candidates.values())
    s1_with_cands = sum(1 for s1 in all_s1_ids if candidates.get(s1))
    avg_cands = unique_pairs / max(s1_with_cands, 1)

    print(f"\n{'='*60}")
    print(f"  Reformatting Complete!")
    print(f"  Total S1 entities:       {len(all_s1_ids):,}")
    print(f"  S1 with candidates:      {s1_with_cands:,}")
    print(f"  Unique candidate pairs:  {unique_pairs:,}")
    print(f"  Avg candidates/entity:   {avg_cands:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
