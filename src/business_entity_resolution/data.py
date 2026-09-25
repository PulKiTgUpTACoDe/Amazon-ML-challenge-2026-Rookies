"""
Data loading utilities for the Business Entity Resolution challenge.

All source files are TSV (tab-separated). Entity IDs are always strings.
"""

import os
from pathlib import Path

import pandas as pd

# ── project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

EXPECTED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLS = ["source1_entity_id", "matched_entity_ids"]


# ── loaders ──────────────────────────────────────────────────────────────
def load_source(path: str | Path) -> pd.DataFrame:
    """Load a source TSV (source1/2/3) and validate schema.

    entity_id is kept as a string. No columns are dropped or renamed.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(EXPECTED_SOURCE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def load_ground_truth(path: str | Path | None = None) -> pd.DataFrame:
    """Load the training ground truth TSV.

    Returns a DataFrame with columns:
        source1_entity_id  (str)
        matched_entity_ids (str — comma-separated, possibly empty)
    """
    if path is None:
        path = TRAIN_DIR / "train_ground_truth.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(EXPECTED_GT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def parse_matched_ids(matched_str: str) -> list[str]:
    """Parse the comma-separated matched_entity_ids string into a list.

    Empty string → empty list.
    """
    if not matched_str or matched_str.strip() == "":
        return []
    return [s.strip() for s in matched_str.split(",") if s.strip()]


# ── convenience loaders ──────────────────────────────────────────────────
def load_train_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_s1, train_s2, train_s3)."""
    return (
        load_source(TRAIN_DIR / "train_source1.tsv"),
        load_source(TRAIN_DIR / "train_source2.tsv"),
        load_source(TRAIN_DIR / "train_source3.tsv"),
    )


def load_test_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (test_s1, test_s2, test_s3)."""
    return (
        load_source(TEST_DIR / "test_source1.tsv"),
        load_source(TEST_DIR / "test_source2.tsv"),
        load_source(TEST_DIR / "test_source3.tsv"),
    )
