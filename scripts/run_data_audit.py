"""
Phase 1 — Data Audit

Run from the project root:
    .venv/Scripts/python scripts/run_data_audit.py

Produces:  artifacts/reports/data_audit.json
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.data import (
    DATASET_DIR,
    TRAIN_DIR,
    TEST_DIR,
    load_source,
    load_ground_truth,
    parse_matched_ids,
)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def profile_source(df, label: str) -> dict:
    """Profile a source dataframe."""
    n = len(df)
    info = {
        "label": label,
        "row_count": n,
        "columns": list(df.columns),
        "entity_id_nunique": df["entity_id"].nunique(),
        "entity_id_duplicates": n - df["entity_id"].nunique(),
    }

    # Missing values
    info["missing_values"] = {
        col: int((df[col] == "").sum()) for col in df.columns
    }

    # Country distribution
    info["countries"] = dict(df["country"].value_counts().items())

    # ID prefix check
    prefixes = df["entity_id"].str[:3].value_counts().to_dict()
    info["id_prefixes"] = prefixes

    # Name length stats
    name_lens = df["business_name"].str.len()
    info["name_length"] = {
        "min": int(name_lens.min()),
        "max": int(name_lens.max()),
        "mean": round(float(name_lens.mean()), 1),
        "median": round(float(name_lens.median()), 1),
        "p5": int(name_lens.quantile(0.05)),
        "p95": int(name_lens.quantile(0.95)),
    }

    # Address length stats
    addr_lens = df["business_address"].str.len()
    info["address_length"] = {
        "min": int(addr_lens.min()),
        "max": int(addr_lens.max()),
        "mean": round(float(addr_lens.mean()), 1),
        "median": round(float(addr_lens.median()), 1),
        "p5": int(addr_lens.quantile(0.05)),
        "p95": int(addr_lens.quantile(0.95)),
    }

    # Empty names / addresses
    info["empty_name_count"] = int((df["business_name"] == "").sum())
    info["empty_address_count"] = int((df["business_address"] == "").sum())

    return info


def profile_ground_truth(gt_df) -> dict:
    """Profile the ground truth file."""
    n = len(gt_df)

    # Parse matches
    match_lists = gt_df["matched_entity_ids"].apply(parse_matched_ids)
    match_counts = match_lists.apply(len)

    info = {
        "total_s1_entities": n,
        "s1_id_nunique": gt_df["source1_entity_id"].nunique(),
        "s1_id_duplicates": n - gt_df["source1_entity_id"].nunique(),
    }

    # Match count distribution
    count_dist = dict(match_counts.value_counts().sort_index().items())
    info["match_count_distribution"] = {int(k): int(v) for k, v in count_dist.items()}

    info["zero_matches"] = int((match_counts == 0).sum())
    info["one_match"] = int((match_counts == 1).sum())
    info["multiple_matches"] = int((match_counts > 1).sum())
    info["singleton_rate"] = round(info["zero_matches"] / n, 4) if n > 0 else 0.0

    info["match_count_stats"] = {
        "min": int(match_counts.min()),
        "max": int(match_counts.max()),
        "mean": round(float(match_counts.mean()), 2),
        "median": round(float(match_counts.median()), 1),
    }

    # Flatten all matched IDs
    all_matched = [mid for lst in match_lists for mid in lst]
    info["total_matched_ids"] = len(all_matched)
    info["unique_matched_ids"] = len(set(all_matched))

    # S2/S3 breakdown
    s2_ids = [m for m in all_matched if m.startswith("S2-")]
    s3_ids = [m for m in all_matched if m.startswith("S3-")]
    info["s2_matched_total"] = len(s2_ids)
    info["s2_matched_unique"] = len(set(s2_ids))
    info["s3_matched_total"] = len(s3_ids)
    info["s3_matched_unique"] = len(set(s3_ids))

    # Reuse: does the same S2/S3 ID appear under multiple S1 entities?
    s2_counter = Counter(s2_ids)
    s3_counter = Counter(s3_ids)
    info["s2_reused_across_s1"] = sum(1 for c in s2_counter.values() if c > 1)
    info["s3_reused_across_s1"] = sum(1 for c in s3_counter.values() if c > 1)
    if info["s2_reused_across_s1"] > 0:
        info["s2_max_reuse"] = max(s2_counter.values())
    if info["s3_reused_across_s1"] > 0:
        info["s3_max_reuse"] = max(s3_counter.values())

    # ID prefix sanity
    bad_prefixes = [m for m in all_matched if not m.startswith(("S2-", "S3-"))]
    info["bad_prefix_matched_ids"] = len(bad_prefixes)

    return info


def main():
    t0 = time.time()
    report = {}

    # ── File sizes ───────────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1 — DATA AUDIT")
    print("=" * 60)

    print("\n--- File Sizes ---")
    file_sizes = {}
    for folder in [TRAIN_DIR, TEST_DIR]:
        for f in sorted(folder.glob("*.tsv")):
            mb = file_size_mb(f)
            rel = f.relative_to(DATASET_DIR)
            file_sizes[str(rel)] = round(mb, 1)
            print(f"  {rel}: {mb:.1f} MB")
    report["file_sizes_mb"] = file_sizes

    # ── Train sources ────────────────────────────────────────────────────
    print("\n--- Loading Train Sources ---")

    t1 = time.time()
    train_s1 = load_source(TRAIN_DIR / "train_source1.tsv")
    print(f"  S1: {len(train_s1):,} rows  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    train_s2 = load_source(TRAIN_DIR / "train_source2.tsv")
    print(f"  S2: {len(train_s2):,} rows  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    train_s3 = load_source(TRAIN_DIR / "train_source3.tsv")
    print(f"  S3: {len(train_s3):,} rows  ({time.time()-t1:.1f}s)")

    report["train_s1"] = profile_source(train_s1, "train_source1")
    report["train_s2"] = profile_source(train_s2, "train_source2")
    report["train_s3"] = profile_source(train_s3, "train_source3")

    # ── Ground truth ─────────────────────────────────────────────────────
    print("\n--- Loading Ground Truth ---")
    t1 = time.time()
    gt = load_ground_truth()
    print(f"  Ground truth: {len(gt):,} rows  ({time.time()-t1:.1f}s)")
    report["ground_truth"] = profile_ground_truth(gt)

    # ── Test sources ─────────────────────────────────────────────────────
    print("\n--- Loading Test Sources ---")

    t1 = time.time()
    test_s1 = load_source(TEST_DIR / "test_source1.tsv")
    print(f"  S1: {len(test_s1):,} rows  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    test_s2 = load_source(TEST_DIR / "test_source2.tsv")
    print(f"  S2: {len(test_s2):,} rows  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    test_s3 = load_source(TEST_DIR / "test_source3.tsv")
    print(f"  S3: {len(test_s3):,} rows  ({time.time()-t1:.1f}s)")

    report["test_s1"] = profile_source(test_s1, "test_source1")
    report["test_s2"] = profile_source(test_s2, "test_source2")
    report["test_s3"] = profile_source(test_s3, "test_source3")

    # ── Cartesian pair estimate ──────────────────────────────────────────
    train_pairs = len(train_s1) * (len(train_s2) + len(train_s3))
    test_pairs = len(test_s1) * (len(test_s2) + len(test_s3))
    report["cartesian"] = {
        "train_s1_x_s2_plus_s3": train_pairs,
        "test_s1_x_s2_plus_s3": test_pairs,
        "train_pairs_billions": round(train_pairs / 1e9, 2),
        "test_pairs_billions": round(test_pairs / 1e9, 2),
    }

    # ── Cross-set checks ────────────────────────────────────────────────
    # Are all ground-truth S1 IDs present in train_source1?
    gt_s1_ids = set(gt["source1_entity_id"])
    train_s1_ids = set(train_s1["entity_id"])
    report["gt_s1_ids_in_train_s1"] = len(gt_s1_ids & train_s1_ids)
    report["gt_s1_ids_missing_from_train_s1"] = len(gt_s1_ids - train_s1_ids)
    report["train_s1_ids_missing_from_gt"] = len(train_s1_ids - gt_s1_ids)

    # ── Summary print ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    report["audit_runtime_seconds"] = round(elapsed, 1)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for split, s1k, s2k, s3k in [
        ("TRAIN", "train_s1", "train_s2", "train_s3"),
        ("TEST", "test_s1", "test_s2", "test_s3"),
    ]:
        s1 = report[s1k]
        s2 = report[s2k]
        s3 = report[s3k]
        print(f"\n{split}:")
        print(f"  S1: {s1['row_count']:>10,}  countries: {s1['countries']}")
        print(f"  S2: {s2['row_count']:>10,}  countries: {s2['countries']}")
        print(f"  S3: {s3['row_count']:>10,}  countries: {s3['countries']}")
        print(f"  S1 empty names: {s1['empty_name_count']}, empty addresses: {s1['empty_address_count']}")
        print(f"  S2 empty names: {s2['empty_name_count']}, empty addresses: {s2['empty_address_count']}")
        print(f"  S3 empty names: {s3['empty_name_count']}, empty addresses: {s3['empty_address_count']}")

    g = report["ground_truth"]
    print(f"\nGROUND TRUTH:")
    print(f"  S1 entities:       {g['total_s1_entities']:,}")
    print(f"  Zero matches:      {g['zero_matches']:,}  ({g['singleton_rate']*100:.1f}%)")
    print(f"  One match:         {g['one_match']:,}")
    print(f"  Multiple matches:  {g['multiple_matches']:,}")
    print(f"  Max matches:       {g['match_count_stats']['max']}")
    print(f"  Mean matches:      {g['match_count_stats']['mean']}")
    print(f"  Total matched IDs: {g['total_matched_ids']:,} (unique: {g['unique_matched_ids']:,})")
    print(f"  S2 reuse across S1: {g['s2_reused_across_s1']:,}")
    print(f"  S3 reuse across S1: {g['s3_reused_across_s1']:,}")
    print(f"  Bad prefix IDs:    {g['bad_prefix_matched_ids']}")

    c = report["cartesian"]
    print(f"\nCARTESIAN PAIR ESTIMATE:")
    print(f"  Train: {c['train_s1_x_s2_plus_s3']:,}  ({c['train_pairs_billions']:.2f} billion)")
    print(f"  Test:  {c['test_s1_x_s2_plus_s3']:,}  ({c['test_pairs_billions']:.2f} billion)")

    print(f"\nCROSS CHECKS:")
    print(f"  GT S1 IDs in train S1:      {report['gt_s1_ids_in_train_s1']:,}")
    print(f"  GT S1 IDs missing from S1:  {report['gt_s1_ids_missing_from_train_s1']}")
    print(f"  Train S1 IDs missing from GT: {report['train_s1_ids_missing_from_gt']}")

    print(f"\nMatch count distribution:")
    for k in sorted(g["match_count_distribution"].keys()):
        v = g["match_count_distribution"][k]
        print(f"  {k} matches: {v:,}")

    print(f"\nAudit runtime: {elapsed:.1f}s")

    # ── Save report ──────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
