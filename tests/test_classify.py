import csv
import json
import os
import pytest

from classify_papers import (
    norm_title, has_abstract, text_completeness, load_and_preprocess,
    paper_block, load_cache, save_to_cache,
    write_india_csv, write_nf_csv, write_excluded_csv, write_social_csv,
    format_funnel, write_funnel,
    count_csv_rows, count_cache_entries,
    NO_ABSTRACT, SOCIAL_FIELDS,
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

# ── paper_block ───────────────────────────────────────────────────────────────

def test_paper_block_full():
    row = {"Title": "Organic farming in Punjab", "Year": "2021",
           "Abstract": "This study examines outcomes.", "Keywords": "organic; Punjab; India"}
    block = paper_block(row)
    assert "Title: Organic farming in Punjab" in block
    assert "Year: 2021" in block
    assert "Abstract: This study examines outcomes." in block
    assert "Keywords: organic; Punjab; India" in block

def test_paper_block_no_keywords_empty_string():
    row = {"Title": "ZBNF study", "Year": "2022",
           "Abstract": "Zero budget farming explored.", "Keywords": ""}
    block = paper_block(row)
    assert "Keywords" not in block

def test_paper_block_none_keywords():
    row = {"Title": "ZBNF study", "Year": "2022",
           "Abstract": "Zero budget farming explored.", "Keywords": None}
    block = paper_block(row)
    assert "Keywords" not in block

def test_paper_block_whitespace_keywords_omitted():
    row = {"Title": "Study", "Year": "2023", "Abstract": "Abstract text.", "Keywords": "   "}
    block = paper_block(row)
    assert "Keywords" not in block

# ── cache ─────────────────────────────────────────────────────────────────────

def test_load_cache_missing_file(tmp_path):
    cache = load_cache(str(tmp_path / "nonexistent.jsonl"))
    assert cache == {}

def test_cache_roundtrip(tmp_path):
    path = str(tmp_path / "cache.jsonl")
    r1 = {"relevant": True, "confidence": "high", "reason": "India is the study site"}
    r2 = {"relevant": False, "confidence": "medium", "reason": "Global study, India not central"}
    save_to_cache(path, "123", r1)
    save_to_cache(path, "456", r2)
    cache = load_cache(path)
    assert "123" in cache
    assert cache["123"]["relevant"] is True
    assert cache["123"]["confidence"] == "high"
    assert cache["123"]["reason"] == "India is the study site"
    assert "456" in cache
    assert cache["456"]["relevant"] is False

def test_cache_appends(tmp_path):
    path = str(tmp_path / "cache.jsonl")
    save_to_cache(path, "1", {"relevant": True, "confidence": "high", "reason": "A"})
    save_to_cache(path, "2", {"relevant": False, "confidence": "low", "reason": "B"})
    cache = load_cache(path)
    assert len(cache) == 2

def test_cache_last_write_wins_on_duplicate_id(tmp_path):
    path = str(tmp_path / "cache.jsonl")
    save_to_cache(path, "1", {"relevant": True, "confidence": "high", "reason": "First"})
    save_to_cache(path, "1", {"relevant": False, "confidence": "low", "reason": "Second"})
    cache = load_cache(path)
    assert cache["1"]["reason"] == "Second"

# ── CSV writers ───────────────────────────────────────────────────────────────

def test_write_india_csv(tmp_path):
    papers = [{
        "Corpus ID": "1", "Authors": "Smith J.", "Year": "2021",
        "Title": "Organic farming in Punjab", "DOI": "10.1234/x",
        "Abstract": "Should not appear",
        "India_confidence": "high", "India_reason": "Punjab is the study site",
    }]
    out = tmp_path / "india.csv"
    write_india_csv(papers, path=str(out))
    rows = list(csv.DictReader(open(str(out))))
    assert len(rows) == 1
    assert rows[0]["India_confidence"] == "high"
    assert rows[0]["India_reason"] == "Punjab is the study site"
    assert "Abstract" not in rows[0]
    assert list(rows[0].keys()) == ["Corpus ID", "Authors", "Year", "Title", "DOI",
                                     "India_confidence", "India_reason"]

def test_write_nf_csv(tmp_path):
    papers = [{
        "Corpus ID": "2", "Authors": "Jones A.", "Year": "2022",
        "Title": "ZBNF in Andhra Pradesh", "DOI": "10.5678/y",
        "India_confidence": "high", "India_reason": "Andhra Pradesh study",
        "NF_confidence": "high", "NF_reason": "ZBNF is the central topic",
    }]
    out = tmp_path / "nf.csv"
    write_nf_csv(papers, path=str(out))
    rows = list(csv.DictReader(open(str(out))))
    assert len(rows) == 1
    assert rows[0]["NF_confidence"] == "high"
    assert rows[0]["NF_reason"] == "ZBNF is the central topic"
    assert list(rows[0].keys()) == ["Corpus ID", "Authors", "Year", "Title", "DOI",
                                     "India_confidence", "India_reason",
                                     "NF_confidence", "NF_reason"]

def test_write_excluded_csv(tmp_path):
    excluded = [
        {"Corpus ID": "3", "Title": "Global agriculture", "Year": "2020",
         "Stage": "duplicate (Title, Year)", "Reason": "Duplicate; richer copy retained"},
        {"Corpus ID": "4", "Title": "Wheat yields in USA", "Year": "2021",
         "Stage": "India screen", "Reason": "Study is set in the United States"},
    ]
    out = tmp_path / "excluded.csv"
    write_excluded_csv(excluded, path=str(out))
    rows = list(csv.DictReader(open(str(out))))
    assert len(rows) == 2
    assert rows[0]["Stage"] == "duplicate (Title, Year)"
    assert rows[1]["Stage"] == "India screen"

def test_write_social_csv(tmp_path):
    papers = [{
        "Corpus ID": "5", "Authors": "Kumar R.", "Year": "2023",
        "Title": "Women and ZBNF in Andhra Pradesh", "DOI": "10.9999/z",
        "India_confidence": "high", "India_reason": "Andhra Pradesh study",
        "NF_confidence": "high", "NF_reason": "ZBNF is the central topic",
        "Abstract": "Should not appear",
        "Social_confidence": "high", "Social_reason": "Gender roles are the central focus",
    }]
    out = tmp_path / "social.csv"
    write_social_csv(papers, path=str(out))
    rows = list(csv.DictReader(open(str(out))))
    assert len(rows) == 1
    assert rows[0]["Social_confidence"] == "high"
    assert rows[0]["Social_reason"] == "Gender roles are the central focus"
    assert "Abstract" not in rows[0]
    assert list(rows[0].keys()) == SOCIAL_FIELDS

def test_count_csv_rows(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    assert count_csv_rows(str(p)) == 2

def test_count_csv_rows_missing_file(tmp_path):
    assert count_csv_rows(str(tmp_path / "missing.csv")) == 0

def test_count_cache_entries(tmp_path):
    p = tmp_path / "cache.jsonl"
    p.write_text('{"corpus_id":"1"}\n{"corpus_id":"2"}\n\n')
    assert count_cache_entries(str(p)) == 2

def test_count_cache_entries_missing_file(tmp_path):
    assert count_cache_entries(str(tmp_path / "missing.jsonl")) == 0

def test_format_funnel_contains_all_counts():
    funnel = {
        "total_read": 100, "removed_title_year": 10, "removed_doi": 5,
        "removed_no_abstract": 15, "to_classify": 70,
        "india_relevant": 20, "india_not_relevant": 48, "india_errors": 2,
        "nf_relevant": 8, "nf_not_relevant": 11, "nf_errors": 1,
    }
    lines = format_funnel(funnel)
    text = "\n".join(lines)
    for expected in ["100", "10", "5", "15", "70", "20", "48", "2", "8", "11", "1"]:
        assert expected in text, f"Expected '{expected}' in funnel summary"
    assert "Social" not in text

def test_format_funnel_includes_social_when_present():
    funnel = {
        "total_read": 100, "removed_title_year": 0, "removed_doi": 0,
        "removed_no_abstract": 0, "to_classify": 100,
        "india_relevant": 40, "india_not_relevant": 60, "india_errors": 0,
        "nf_relevant": 20, "nf_not_relevant": 20, "nf_errors": 0,
        "social_relevant": 7, "social_not_relevant": 13, "social_errors": 0,
    }
    text = "\n".join(format_funnel(funnel))
    assert "Social dimensions shortlist" in text
    assert "7" in text
    assert "13" in text

def test_write_funnel_writes_file_and_prints(tmp_path, capsys):
    funnel = {
        "total_read": 10, "removed_title_year": 1, "removed_doi": 0,
        "removed_no_abstract": 2, "to_classify": 7,
        "india_relevant": 3, "india_not_relevant": 4, "india_errors": 0,
        "nf_relevant": 2, "nf_not_relevant": 1, "nf_errors": 0,
    }
    out = tmp_path / "funnel.txt"
    write_funnel(funnel, path=str(out))
    assert out.exists()
    content = out.read_text()
    assert "10" in content
    captured = capsys.readouterr()
    assert "10" in captured.out
