import csv
import json
import os
import pytest

from classify_papers import (
    norm_title, has_abstract, text_completeness, load_and_preprocess,
    paper_block, load_cache, save_to_cache,
    write_india_csv, write_nf_csv, write_excluded_csv,
    format_funnel, write_funnel,
    NO_ABSTRACT,
)

FIELDS = ["Corpus ID", "Authors", "Year", "Title", "Abstract", "Keywords",
          "Source", "DOI", "Document Type", "Database"]

def make_row(**kwargs):
    defaults = {f: "" for f in FIELDS}
    defaults.update(kwargs)
    return defaults

def write_fixture_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

# ── norm_title ────────────────────────────────────────────────────────────────

def test_norm_title_lowercases():
    assert norm_title("Hello World") == "hello world"

def test_norm_title_collapses_whitespace():
    assert norm_title("  hello   world  ") == "hello world"

def test_norm_title_none():
    assert norm_title(None) == ""

def test_norm_title_empty():
    assert norm_title("") == ""

# ── has_abstract ──────────────────────────────────────────────────────────────

def test_has_abstract_empty_string():
    assert not has_abstract({"Abstract": ""})

def test_has_abstract_sentinel():
    assert not has_abstract({"Abstract": NO_ABSTRACT})

def test_has_abstract_whitespace_only():
    assert not has_abstract({"Abstract": "   "})

def test_has_abstract_real():
    assert has_abstract({"Abstract": "This paper studies organic farming in India."})

# ── text_completeness ─────────────────────────────────────────────────────────

def test_text_completeness_sums_lengths():
    assert text_completeness({"Abstract": "abc", "Keywords": "de"}) == 5

def test_text_completeness_sentinel_counts_zero():
    assert text_completeness({"Abstract": NO_ABSTRACT, "Keywords": "de"}) == 2

def test_text_completeness_none_keywords():
    assert text_completeness({"Abstract": "abc", "Keywords": None}) == 3

def test_text_completeness_empty_abstract():
    assert text_completeness({"Abstract": "", "Keywords": "kw"}) == 2
