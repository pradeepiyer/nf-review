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

# ── load_and_preprocess ───────────────────────────────────────────────────────

def test_dedup_title_year_keeps_richer(tmp_path):
    rows = [
        make_row(**{"Corpus ID": "1", "Title": "Farming in India", "Year": "2020",
                    "Abstract": "Short", "DOI": "10.1/a"}),
        make_row(**{"Corpus ID": "2", "Title": "FARMING IN INDIA", "Year": "2020",
                    "Abstract": "Much longer abstract text", "Keywords": "kw1", "DOI": "10.1/b"}),
        make_row(**{"Corpus ID": "3", "Title": "Another Paper", "Year": "2021",
                    "Abstract": "Good abstract", "DOI": "10.1/c"}),
    ]
    p = tmp_path / "papers.csv"
    write_fixture_csv(str(p), rows)
    papers, funnel, excluded = load_and_preprocess(str(p))
    assert funnel["total_read"] == 3
    assert funnel["removed_title_year"] == 1
    assert any(r["Corpus ID"] == "2" for r in papers)   # richer copy kept
    assert not any(r["Corpus ID"] == "1" for r in papers)
    assert any(e["Corpus ID"] == "1" and e["Stage"] == "duplicate (Title, Year)" for e in excluded)


def test_dedup_doi_keeps_richer(tmp_path):
    rows = [
        make_row(**{"Corpus ID": "1", "Title": "Paper A", "Year": "2020",
                    "Abstract": "Short abstract", "DOI": "10.1/x"}),
        make_row(**{"Corpus ID": "2", "Title": "Paper B", "Year": "2019",
                    "Abstract": "Much longer abstract here", "Keywords": "kw1", "DOI": "10.1/x"}),
        make_row(**{"Corpus ID": "3", "Title": "Paper C", "Year": "2021",
                    "Abstract": "Good abstract", "DOI": ""}),  # no-DOI row always kept
    ]
    p = tmp_path / "papers.csv"
    write_fixture_csv(str(p), rows)
    papers, funnel, excluded = load_and_preprocess(str(p))
    assert funnel["removed_title_year"] == 0
    assert funnel["removed_doi"] == 1
    assert any(r["Corpus ID"] == "2" for r in papers)   # richer kept
    assert not any(r["Corpus ID"] == "1" for r in papers)
    assert any(r["Corpus ID"] == "3" for r in papers)   # no-DOI kept
    assert any(e["Corpus ID"] == "1" and e["Stage"] == "duplicate (DOI)" for e in excluded)


def test_removes_no_abstract(tmp_path):
    rows = [
        make_row(**{"Corpus ID": "1", "Title": "Paper A", "Year": "2020",
                    "Abstract": "Good abstract", "DOI": "10.1/a"}),
        make_row(**{"Corpus ID": "2", "Title": "Paper B", "Year": "2021",
                    "Abstract": NO_ABSTRACT, "DOI": "10.1/b"}),
        make_row(**{"Corpus ID": "3", "Title": "Paper C", "Year": "2022",
                    "Abstract": "", "DOI": "10.1/c"}),
    ]
    p = tmp_path / "papers.csv"
    write_fixture_csv(str(p), rows)
    papers, funnel, excluded = load_and_preprocess(str(p))
    assert funnel["removed_no_abstract"] == 2
    assert funnel["to_classify"] == 1
    assert papers[0]["Corpus ID"] == "1"
    assert any(e["Corpus ID"] == "2" and e["Stage"] == "no abstract" for e in excluded)
    assert any(e["Corpus ID"] == "3" and e["Stage"] == "no abstract" for e in excluded)


def test_funnel_totals_consistent(tmp_path):
    rows = [
        # row 1 + 2: same (title, year) → row 2 richer (longer abstract)
        make_row(**{"Corpus ID": "1", "Title": "Paper A", "Year": "2020",
                    "Abstract": "Short", "DOI": "10.1/a"}),
        make_row(**{"Corpus ID": "2", "Title": "Paper A", "Year": "2020",
                    "Abstract": "Longer abstract here", "DOI": "10.1/b"}),
        # row 3 + 4: different titles, same DOI → row 4 richer
        make_row(**{"Corpus ID": "3", "Title": "Paper C", "Year": "2021",
                    "Abstract": "Short C", "DOI": "10.1/c"}),
        make_row(**{"Corpus ID": "4", "Title": "Paper D", "Year": "2022",
                    "Abstract": "Longer abstract D", "DOI": "10.1/c"}),
        # row 5: no abstract → dropped
        make_row(**{"Corpus ID": "5", "Title": "Paper E", "Year": "2023",
                    "Abstract": "", "DOI": "10.1/e"}),
        # row 6: passes all stages
        make_row(**{"Corpus ID": "6", "Title": "Paper F", "Year": "2024",
                    "Abstract": "Abstract F", "DOI": "10.1/f"}),
    ]
    p = tmp_path / "papers.csv"
    write_fixture_csv(str(p), rows)
    papers, funnel, excluded = load_and_preprocess(str(p))
    assert funnel["total_read"] == 6
    assert funnel["removed_title_year"] == 1   # row 1 (row 2 is richer)
    assert funnel["removed_doi"] == 1           # row 3 (row 4 is richer)
    assert funnel["removed_no_abstract"] == 1   # row 5
    assert funnel["to_classify"] == 3           # rows 2, 4, 6
    assert len(papers) == 3
    assert len(excluded) == 3
    stages = {e["Corpus ID"]: e["Stage"] for e in excluded}
    assert stages["1"] == "duplicate (Title, Year)"
    assert stages["3"] == "duplicate (DOI)"
    assert stages["5"] == "no abstract"
