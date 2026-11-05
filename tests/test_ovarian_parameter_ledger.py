"""Offline tests for the ovarian HGSOC parameter ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_ovarian_parameter_ledger import (
    LEDGER_PATH,
    check_dois,
    label_counts,
    load_ledger,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ledger_file_exists():
    assert LEDGER_PATH.is_file()


def test_schema_validation_passes():
    ledger = load_ledger()
    errors = validate_schema(ledger)
    assert errors == []


def test_required_candidate_quantities_represented():
    ledger = load_ledger()
    text = json.dumps(ledger).casefold()
    needles = [
        "ca-125",
        "doubling time",
        "platinum-free interval",
        "response rate",
        "residual",
        "brca",
        "hrd",
        "recurrence",
        "parp",
        "olaparib",
        "carboplatin",
        "paclitaxel",
    ]
    missing = [n for n in needles if n not in text]
    assert not missing, f"missing topic needles: {missing}"


def test_missing_rows_have_null_value():
    ledger = load_ledger()
    for entry in ledger["entries"]:
        if entry["label"] == "MISSING":
            assert entry["value"] is None


def test_label_counts_stable():
    counts = label_counts(load_ledger())
    assert counts["KNOWN"] >= 10
    assert counts["MISSING"] >= 5
    assert sum(counts.values()) == len(load_ledger()["entries"])


@pytest.mark.network
def test_crossref_doi_check_live():
    report = check_dois(load_ledger())
    assert report["failures"] == 0
    assert report["unique_dois_checked"] >= 10
