"""
Improved multi-strategy candidate generation for business entity resolution.

Blocking Strategies (union of all):
  1. TF-IDF char n-grams on business_name  — fuzzy name matching
  2. TF-IDF char n-grams on business_address — same-address matching
  3. Exact match on normalized business_name — trivial matches

Outputs:
  output/candidate_pairs_raw.tsv  — internal one-pair-per-row (for ML scoring)
  output/candidate_pairs.tsv      — competition grouped format (for submission)
"""
import sys
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from business_entity_resolution.data import load_source, TRAIN_DIR, TEST_DIR
from business_entity_resolution.normalization import normalize_dataframe


# ── Blocking Functions ──────────────────────────────────────────────────────

def tfidf_blocking_to_file(X1, s1_ids, target_chunk, vectorizer, out_file,
                            column="business_name", top_k=10, threshold=0.3):
    """
    TF-IDF sparse dot-product blocking using sparse_dot_topn (C++).
    X1 is precomputed for S1 and reused across target chunks.
    Appends pairs to *out_file*.
    """
    from sparse_dot_topn import sp_matmul_topn

    target_texts = target_chunk[column].fillna("").astype(str)
    X2 = vectorizer.transform(target_texts)
    X2_T = sp.csr_matrix(X2).T

    target_ids = target_chunk["entity_id"].values
    total_written = 0
    s1_chunk_size = 100_000  # chunk X1 to limit RAM

    with open(out_file, "a", encoding="utf-8") as f:
        for start in range(0, X1.shape[0], s1_chunk_size):
            end = min(start + s1_chunk_size, X1.shape[0])
            X1_chunk = sp.csr_matrix(X1[start:end])

            res = sp_matmul_topn(X1_chunk, X2_T, top_n=top_k, threshold=threshold)

            lines = []
            for i in range(res.shape[0]):
                s_ptr, e_ptr = res.indptr[i], res.indptr[i + 1]
                if s_ptr == e_ptr:
                    continue
                s1_id = s1_ids[start + i]
                for t_idx in res.indices[s_ptr:e_ptr]:
                    lines.append(f"{s1_id}\t{target_ids[t_idx]}\n")

            if lines:
                f.writelines(lines)
                total_written += len(lines)

    return total_written


def exact_match_blocking_to_file(s1_name_index, target_chunk, out_file,
                                  column="business_name"):
    """Exact match on normalized name using pre-built inverted index."""
    lines = []
    for name, t_id in zip(
        target_chunk[column].fillna("").astype(str).values,
        target_chunk["entity_id"].values,
    ):
        if name and len(name) > 2 and name in s1_name_index:
            for s1_id in s1_name_index[name]:
                lines.append(f"{s1_id}\t{t_id}\n")

    if lines:
        with open(out_file, "a", encoding="utf-8") as f:
            f.writelines(lines)
    return len(lines)


# ── Format Conversion ────────────────────────────────────────────────────────

def convert_raw_to_competition(raw_file, out_file, all_s1_ids):
    """Stream raw pairs → deduplicate → group by S1 → competition TSV."""
    print("\nStep 6: Converting raw pairs to competition format (deduplicate + group)...")
    t0 = time.time()

    candidates = defaultdict(set)
    chunk_iter = pd.read_csv(raw_file, sep="\t", chunksize=5_000_000, dtype=str)

    for i, chunk in enumerate(chunk_iter):
        col_s1, col_t = chunk.columns[0], chunk.columns[1]
        for s1_id, t_id in zip(chunk[col_s1], chunk[col_t]):
            if pd.notna(s1_id) and pd.notna(t_id):
                candidates[s1_id].add(t_id)
        print(f"  Dedup chunk {i+1}...")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, set())
            joined = ",".join(sorted(cands)) if cands else ""
            f.write(f"{s1_id}\t{joined}\n")

    unique = sum(len(v) for v in candidates.values())
    with_cands = sum(1 for s in all_s1_ids if candidates.get(s))
    print(f"  {unique:,} unique pairs, {with_cands:,}/{len(all_s1_ids):,} S1 have candidates. ({time.time()-t0:.1f}s)")
    return unique


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-strategy candidate generation")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = PROJECT_ROOT / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    data_dir = TEST_DIR if args.mode == "test" else TRAIN_DIR
    raw_file = str(OUTPUT_DIR / "candidate_pairs_raw.tsv")
    competition_file = str(OUTPUT_DIR / "candidate_pairs.tsv")
    cols = ["entity_id", "business_name", "business_address", "country"]

    # Clear raw file
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write("s1_id\ttarget_id\n")

    # ── Step 1: Load and normalize S1 ──
    print(f"Step 1: Loading + normalizing S1 ({args.mode})...")
    t0 = time.time()
    s1 = load_source(data_dir / f"{args.mode}_source1.tsv", usecols=cols)
    s1 = normalize_dataframe(s1, cols)
    all_s1_ids = s1["entity_id"].values.tolist()
    print(f"  {len(s1):,} S1 entities in {time.time()-t0:.1f}s")

    # ── Step 2: Fit TF-IDF vectorizers on NORMALIZED text ──
    print("Step 2: Fitting TF-IDF vectorizers on normalized S1 text...")
    t0 = time.time()

    name_texts = s1["business_name"].fillna("").astype(str)
    name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_df=0.01)
    sample_n = min(500_000, len(name_texts))
    name_vec.fit(name_texts.sample(n=sample_n, random_state=42))
    print(f"  Name vectorizer: {len(name_vec.vocabulary_):,} char n-gram features")

    addr_texts = s1["business_address"].fillna("").astype(str)
    s1_has_addr_mask = addr_texts.str.len() > 3
    has_addr_vec = False
    addr_vec = None
    if s1_has_addr_mask.sum() > 1000:
        addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_df=0.01)
        addr_vec.fit(
            addr_texts[s1_has_addr_mask].sample(
                n=min(500_000, int(s1_has_addr_mask.sum())), random_state=42
            )
        )
        has_addr_vec = True
        print(f"  Addr vectorizer: {len(addr_vec.vocabulary_):,} char n-gram features")
    else:
        print("  Addr vectorizer: SKIPPED (insufficient address data)")

    print(f"  Fitted in {time.time()-t0:.1f}s")

    # ── Step 3: Precompute S1 TF-IDF matrices (reused across ALL target chunks) ──
    print("Step 3: Precomputing S1 sparse matrices (done once)...")
    t0 = time.time()

    X1_name = sp.csr_matrix(name_vec.transform(name_texts))
    s1_ids_name = s1["entity_id"].values
    print(f"  X1_name: {X1_name.shape}, nnz={X1_name.nnz:,}")

    X1_addr = None
    s1_ids_addr = None
    if has_addr_vec:
        s1_with_addr = s1[s1_has_addr_mask]
        X1_addr = sp.csr_matrix(
            addr_vec.transform(s1_with_addr["business_address"].fillna("").astype(str))
        )
        s1_ids_addr = s1_with_addr["entity_id"].values
        print(f"  X1_addr: {X1_addr.shape}, nnz={X1_addr.nnz:,}")

    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Step 4: Build exact-match inverted index ──
    print("Step 4: Building exact-match name index...")
    s1_name_index = defaultdict(list)
    for eid, name in zip(s1["entity_id"].values, name_texts.values):
        if name and len(name) > 2:
            s1_name_index[name].append(eid)
    print(f"  {len(s1_name_index):,} unique normalized names indexed")

    # ── Step 5: Process S2 and S3 in 1 M-row chunks ──
    total_raw = 0

    for source_idx in [2, 3]:
        fname = f"{args.mode}_source{source_idx}.tsv"
        fpath = data_dir / fname
        print(f"\n{'='*60}")
        print(f"  Processing: {fname}")
        print(f"{'='*60}")

        chunk_iter = pd.read_csv(
            fpath, sep="\t", usecols=cols, dtype=str,
            keep_default_na=False, chunksize=1_000_000,
        )

        for ci, target_chunk in enumerate(chunk_iter):
            print(f"\n  ── Target Chunk {ci+1} ({len(target_chunk):,} rows) ──")
            target_chunk = normalize_dataframe(target_chunk, cols)

            # Strategy 1: Name TF-IDF  (primary — catches fuzzy name matches)
            print("    [1/3] Name TF-IDF (top_k=10, threshold=0.3)...")
            t0 = time.time()
            n1 = tfidf_blocking_to_file(
                X1_name, s1_ids_name, target_chunk, name_vec,
                raw_file, column="business_name", top_k=10, threshold=0.3,
            )
            total_raw += n1
            print(f"          → {n1:,} pairs ({time.time()-t0:.1f}s)")

            # Strategy 2: Address TF-IDF  (catches same-address different-name)
            if has_addr_vec and X1_addr is not None:
                tgt_has_addr = target_chunk["business_address"].fillna("").str.len() > 3
                if tgt_has_addr.sum() > 100:
                    print("    [2/3] Addr TF-IDF (top_k=5, threshold=0.35)...")
                    t0 = time.time()
                    n2 = tfidf_blocking_to_file(
                        X1_addr, s1_ids_addr,
                        target_chunk[tgt_has_addr], addr_vec,
                        raw_file, column="business_address", top_k=5, threshold=0.35,
                    )
                    total_raw += n2
                    print(f"          → {n2:,} pairs ({time.time()-t0:.1f}s)")
                else:
                    print("    [2/3] Addr TF-IDF: skipped (no addresses in chunk)")
            else:
                print("    [2/3] Addr TF-IDF: skipped")

            # Strategy 3: Exact match  (catches trivially identical names)
            print("    [3/3] Exact match...")
            t0 = time.time()
            n3 = exact_match_blocking_to_file(s1_name_index, target_chunk, raw_file)
            total_raw += n3
            print(f"          → {n3:,} pairs ({time.time()-t0:.1f}s)")

            del target_chunk  # free memory before next chunk

    print(f"\n{'='*60}")
    print(f"  Raw candidate generation complete: {total_raw:,} pairs (pre-dedup)")
    print(f"{'='*60}")

    # ── Step 6: Deduplicate and write competition format ──
    convert_raw_to_competition(raw_file, competition_file, all_s1_ids)

    print(f"\nDone! Pipeline complete!")
    print(f"  Raw pairs:          {raw_file}")
    print(f"  Competition format: {competition_file}")


if __name__ == "__main__":
    main()
