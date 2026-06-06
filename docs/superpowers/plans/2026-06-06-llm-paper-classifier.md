# LLM Paper Classifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the keyword-scoring pipeline with a two-step LLM classifier that filters papers first for substantive India relevance, then for alternative/sustainable farming content, producing an auditable PRISMA-style funnel.

**Architecture:** A single async Python script (`classify_papers.py`) loads and deduplicates the corpus, filters out no-abstract papers, then fans out two sequential LLM classification sweeps (India → natural farming) via `AsyncOpenAI` with bounded concurrency and JSONL caching for resumability. Every exclusion at every stage is recorded in `excluded_papers.csv`; final counts are printed to stdout and saved to `funnel_summary.txt`.

**Tech Stack:** Python 3.11+, `openai` (AsyncOpenAI), `asyncio`, `csv`, `json`, `argparse` — stdlib only beyond `openai`. Run via `uv run classify_papers.py`. Tests run via `uv run --with pytest --with openai pytest tests/ -v`.

---

## File Map

| Path | Status | Responsibility |
|------|--------|----------------|
| `classify_papers.py` | **Create** | Single-script pipeline: config, preprocessing, classify, run_step, CSV writers, main |
| `tests/test_classify.py` | **Create** | Unit tests for all pure/IO functions (no API mocking) |
| `conftest.py` | **Create** | Root-level conftest so pytest finds `classify_papers` on sys.path |
| `README.md` | **Rewrite** | Document the new LLM pipeline |
| `.gitignore` | **Modify** | Add `cache_*.jsonl` |
| `score_papers.py` | **Delete** | Old keyword scorer |
| `make_sample.py` | **Delete** | Old sample generator |
| `Terms.csv` | **Delete** | Old vocabulary file |
| `paper_scores_ranked.csv` | **Delete** | Old output |
| `top_100.csv` | **Delete** | Old output |
| `sample_papers.csv` | **Delete** | Old output |

---

## Task 1: Repo cleanup

**Files:**
- Delete: `score_papers.py`, `make_sample.py`, `Terms.csv`, `paper_scores_ranked.csv`, `top_100.csv`, `sample_papers.csv`
- Modify: `.gitignore`

- [ ] **Step 1: Delete old keyword-method artifacts**

```bash
git rm score_papers.py make_sample.py Terms.csv paper_scores_ranked.csv top_100.csv sample_papers.csv
```

Expected: each file listed as `deleted`.

- [ ] **Step 2: Add cache and output files to .gitignore**

Open `.gitignore` and append:

```
# LLM classifier transient caches
cache_*.jsonl
```

The output CSVs (`india_papers.csv`, `natural_farming_papers.csv`, `excluded_papers.csv`, `funnel_summary.txt`) are bibliographic-only and **safe to commit** — do NOT gitignore them.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "Remove keyword-scoring artifacts; gitignore classifier caches"
```

---

## Task 2: Script skeleton + test setup

**Files:**
- Create: `classify_papers.py`
- Create: `tests/test_classify.py`
- Create: `conftest.py`

- [ ] **Step 1: Create `conftest.py` at the repo root**

```python
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 2: Create `classify_papers.py` with PEP 723 header, all config, prompts, and stub functions**

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai"]
# ///

import argparse
import asyncio
import csv
import json
import os

import openai
from openai import AsyncOpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL = "gpt-5-mini"
PAPERS_FILE = "Total_Paper_List.csv"
INDIA_CACHE = "cache_india.jsonl"
NF_CACHE = "cache_natural_farming.jsonl"
INDIA_OUT = "india_papers.csv"
NF_OUT = "natural_farming_papers.csv"
EXCLUDED_OUT = "excluded_papers.csv"
FUNNEL_OUT = "funnel_summary.txt"
NO_ABSTRACT = "[No abstract available]"
CONCURRENCY = 10
MAX_RETRIES = 5
BASE_DELAY = 1.0

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {"type": "boolean"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reason": {"type": "string"},
            },
            "required": ["relevant", "confidence", "reason"],
            "additionalProperties": False,
        },
    },
}

INDIA_PROMPT = """\
You are classifying academic papers for a systematic review of Indian agriculture and rural development.

Your task: Determine whether this paper is SUBSTANTIVELY about India. India (or a specific Indian state, region, or city) must be a CENTRAL focus — the primary study site, the policy context studied, the population of interest, the main dataset, or the primary case study. A paper that merely mentions India in passing, or lists it alongside many other countries without India-specific findings, is NOT relevant.

Examples:
- "Land reform in Maharashtra: A case study of smallholder transitions" → relevant (Indian state is the study site)
- "Smallholder agriculture across South and Southeast Asia: Vietnam, Thailand, and India" → not relevant (India is one of many countries, not the central focus)
- "Andhra Pradesh's Community-managed Natural Farming programme: outcomes and lessons" → relevant (specific Indian state is the focus)
- "Global trends in organic farming adoption, 2000–2020" → not relevant (global study, India not featured centrally)
- "How Indian farmers respond to price shocks: panel evidence from ICRISAT villages" → relevant (India-specific dataset and population)

Respond with a JSON object:
- relevant: true if India is a central focus, false otherwise
- confidence: "high" if clearly one way or the other, "medium" if uncertain, "low" if information is very thin
- reason: one sentence describing your judgment — do NOT quote the abstract; describe your reasoning generically
"""

NF_PROMPT = """\
You are classifying academic papers for a systematic review of alternative and sustainable farming in India. These papers have already been confirmed to be about India.

Your task: Determine whether this paper addresses any NON-CONVENTIONAL, SUSTAINABLE, or AGROECOLOGICAL farming approach. Include:
- Natural farming / Zero Budget Natural Farming (ZBNF) / Subhash Palekar Natural Farming (SPNF)
- Andhra Pradesh Community-managed Natural Farming (APCNF)
- Organic farming, agroecology, permaculture, regenerative agriculture
- Conservation agriculture, biodynamic farming
- Traditional or indigenous farming practices
- Low-external-input or chemical-free farming
- Any approach framed as an alternative to conventional/industrial agriculture

Papers about PURELY conventional or industrial agriculture with no alternative-farming angle are NOT relevant.

Examples:
- "Transition to organic cotton farming in Gujarat: farmer outcomes" → relevant
- "ZBNF adoption among marginal farmers in Andhra Pradesh" → relevant
- "Agroecological practices and biodiversity in Indian rice systems" → relevant
- "Yield response of wheat to nitrogen fertilizer in Punjab" → not relevant (conventional agronomy)
- "Indigenous seed saving practices in tribal communities of Odisha" → relevant (traditional farming)
- "Green revolution adoption patterns in Indian states, 1965–1985" → not relevant (conventional intensification)

Respond with a JSON object:
- relevant: true if any alternative/sustainable/agroecological farming angle is present, false if purely conventional
- confidence: "high" if clear, "medium" if uncertain, "low" if focus is hard to determine
- reason: one sentence describing your judgment — do NOT quote the abstract
"""

# ── OUTPUT SCHEMAS ────────────────────────────────────────────────────────────

INDIA_FIELDS = ["Corpus ID", "Authors", "Year", "Title", "DOI", "India_confidence", "India_reason"]
NF_FIELDS = ["Corpus ID", "Authors", "Year", "Title", "DOI", "India_confidence", "India_reason", "NF_confidence", "NF_reason"]
EXCLUDED_FIELDS = ["Corpus ID", "Title", "Year", "Stage", "Reason"]

# ── PREPROCESSING ─────────────────────────────────────────────────────────────

def norm_title(title):
    pass

def has_abstract(row):
    pass

def text_completeness(row):
    pass

def load_and_preprocess(path=PAPERS_FILE):
    pass

# ── TEXT ASSEMBLY ─────────────────────────────────────────────────────────────

def paper_block(row):
    pass

# ── CACHE ─────────────────────────────────────────────────────────────────────

def load_cache(path):
    pass

def save_to_cache(path, corpus_id, result):
    pass

# ── CSV WRITERS + FUNNEL ──────────────────────────────────────────────────────

def write_india_csv(papers, path=INDIA_OUT):
    pass

def write_nf_csv(papers, path=NF_OUT):
    pass

def write_excluded_csv(excluded, path=EXCLUDED_OUT):
    pass

def format_funnel(funnel):
    pass

def write_funnel(funnel, path=FUNNEL_OUT):
    pass

# ── ASYNC CLASSIFIER ──────────────────────────────────────────────────────────

async def classify(client, block, system_prompt, semaphore):
    pass

async def run_step(client, papers, system_prompt, cache_path, step_label):
    pass

# ── MAIN ──────────────────────────────────────────────────────────────────────

async def async_main(smoke=False):
    pass

def main():
    parser = argparse.ArgumentParser(description="Two-step LLM classifier for academic papers.")
    parser.add_argument("--smoke", action="store_true", help="Run on first 3 papers only to verify API and estimate cost.")
    args = parser.parse_args()
    asyncio.run(async_main(smoke=args.smoke))

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `tests/test_classify.py` with imports and fixture helpers**

```python
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
```

- [ ] **Step 4: Verify import succeeds**

```bash
uv run --with pytest --with openai python -c "import classify_papers; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit skeleton**

```bash
git add classify_papers.py tests/test_classify.py conftest.py
git commit -m "Add script skeleton, test file, and conftest"
```

---

## Task 3: TDD — Preprocessing helpers

**Files:**
- Modify: `tests/test_classify.py` (add tests)
- Modify: `classify_papers.py` (implement `norm_title`, `has_abstract`, `text_completeness`)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_classify.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "norm_title or has_abstract or text_completeness" -v
```

Expected: multiple FAILED with `TypeError` or `AssertionError` (functions return `None` from `pass`).

- [ ] **Step 3: Implement `norm_title`, `has_abstract`, `text_completeness`**

In `classify_papers.py`, replace the three stubs:

```python
def norm_title(title):
    return " ".join((title or "").strip().lower().split())

def has_abstract(row):
    a = (row.get("Abstract") or "").strip()
    return a != "" and a != NO_ABSTRACT

def text_completeness(row):
    abstract = "" if (row.get("Abstract") or "").strip() == NO_ABSTRACT else (row.get("Abstract") or "")
    return len(abstract) + len(row.get("Keywords") or "")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "norm_title or has_abstract or text_completeness" -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add classify_papers.py tests/test_classify.py
git commit -m "Implement and test preprocessing helpers"
```

---

## Task 4: TDD — load_and_preprocess

**Files:**
- Modify: `tests/test_classify.py`
- Modify: `classify_papers.py`

`load_and_preprocess(path) -> tuple[list[dict], dict, list[dict]]` returns:
- `papers`: list of paper row dicts that survived all three preprocessing stages
- `funnel`: dict with keys `total_read`, `removed_title_year`, `removed_doi`, `removed_no_abstract`, `to_classify`
- `excluded`: list of dicts with keys `Corpus ID`, `Title`, `Year`, `Stage`, `Reason`, one per rejected paper

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_classify.py`:

```python
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
    assert funnel["removed_title_year"] == 1   # row 1
    assert funnel["removed_doi"] == 1           # row 3
    assert funnel["removed_no_abstract"] == 1   # row 5
    assert funnel["to_classify"] == 3           # rows 2, 4, 6
    assert len(papers) == 3
    assert len(excluded) == 3
    stages = {e["Corpus ID"]: e["Stage"] for e in excluded}
    assert stages["1"] == "duplicate (Title, Year)"
    assert stages["3"] == "duplicate (DOI)"
    assert stages["5"] == "no abstract"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "load_and_preprocess or dedup or removes_no or funnel" -v
```

Expected: all FAILED (`TypeError` from `None` return on `pass`).

- [ ] **Step 3: Implement `load_and_preprocess`**

In `classify_papers.py`, replace the `load_and_preprocess` stub:

```python
def load_and_preprocess(path=PAPERS_FILE):
    csv.field_size_limit(10 ** 8)
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)

    # Stage 1: dedup by (Title, Year) — keep richest copy
    best_ty = {}
    for row in rows:
        key = (norm_title(row["Title"]), (row.get("Year") or "").strip())
        if key not in best_ty or text_completeness(row) > text_completeness(best_ty[key]):
            best_ty[key] = row
    after_ty = list(best_ty.values())
    removed_ty = total - len(after_ty)

    # Stage 2: dedup by DOI — keep richest copy; no-DOI rows always kept
    by_doi = {}
    no_doi = []
    for row in after_ty:
        doi = (row.get("DOI") or "").strip().lower()
        if not doi:
            no_doi.append(row)
        elif doi not in by_doi or text_completeness(row) > text_completeness(by_doi[doi]):
            by_doi[doi] = row
    after_doi = list(by_doi.values()) + no_doi
    removed_doi = len(after_ty) - len(after_doi)

    # Stage 3: remove papers with no abstract
    with_abs = [r for r in after_doi if has_abstract(r)]
    removed_noabs = len(after_doi) - len(with_abs)

    # Build excluded rows, attributing each dropped row to its earliest exclusion stage
    ty_ids = {r["Corpus ID"] for r in after_ty}
    doi_ids = {r["Corpus ID"] for r in after_doi}
    abs_ids = {r["Corpus ID"] for r in with_abs}

    excluded = []
    seen = set()
    for row in rows:
        cid = row["Corpus ID"]
        if cid in seen:
            continue
        if cid not in ty_ids:
            seen.add(cid)
            excluded.append({"Corpus ID": cid, "Title": row["Title"],
                             "Year": row.get("Year", ""), "Stage": "duplicate (Title, Year)",
                             "Reason": "Duplicate title and year; richer copy retained"})
        elif cid not in doi_ids:
            seen.add(cid)
            excluded.append({"Corpus ID": cid, "Title": row["Title"],
                             "Year": row.get("Year", ""), "Stage": "duplicate (DOI)",
                             "Reason": "Duplicate DOI; richer copy retained"})
        elif cid not in abs_ids:
            seen.add(cid)
            excluded.append({"Corpus ID": cid, "Title": row["Title"],
                             "Year": row.get("Year", ""), "Stage": "no abstract",
                             "Reason": "No abstract available"})

    funnel = {
        "total_read": total,
        "removed_title_year": removed_ty,
        "removed_doi": removed_doi,
        "removed_no_abstract": removed_noabs,
        "to_classify": len(with_abs),
    }
    return with_abs, funnel, excluded
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "load_and_preprocess or dedup or removes_no or funnel" -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add classify_papers.py tests/test_classify.py
git commit -m "Implement and test load_and_preprocess with three-stage dedup + funnel"
```

---

## Task 5: TDD — paper_block

**Files:**
- Modify: `tests/test_classify.py`
- Modify: `classify_papers.py`

`paper_block(row) -> str`: returns a labeled text block for the model. Keywords are omitted when empty.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_classify.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "paper_block" -v
```

Expected: all FAILED.

- [ ] **Step 3: Implement `paper_block`**

```python
def paper_block(row):
    parts = [
        f"Title: {row['Title']}",
        f"Year: {row.get('Year', '')}",
        f"Abstract: {row['Abstract']}",
    ]
    kw = (row.get("Keywords") or "").strip()
    if kw:
        parts.append(f"Keywords: {kw}")
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "paper_block" -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add classify_papers.py tests/test_classify.py
git commit -m "Implement and test paper_block"
```

---

## Task 6: TDD — cache

**Files:**
- Modify: `tests/test_classify.py`
- Modify: `classify_papers.py`

`load_cache(path) -> dict`: reads JSONL, returns `{corpus_id: {relevant, confidence, reason}}`. Returns `{}` for missing file.

`save_to_cache(path, corpus_id, result)`: appends one line to the JSONL file. `result` is `{relevant, confidence, reason}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_classify.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "cache" -v
```

Expected: all FAILED.

- [ ] **Step 3: Implement `load_cache` and `save_to_cache`**

```python
def load_cache(path):
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                cache[entry["corpus_id"]] = {
                    "relevant": entry["relevant"],
                    "confidence": entry["confidence"],
                    "reason": entry["reason"],
                }
    return cache

def save_to_cache(path, corpus_id, result):
    entry = {"corpus_id": corpus_id, **result}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "cache" -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add classify_papers.py tests/test_classify.py
git commit -m "Implement and test cache (load_cache, save_to_cache)"
```

---

## Task 7: TDD — CSV writers and funnel formatter

**Files:**
- Modify: `tests/test_classify.py`
- Modify: `classify_papers.py`

`write_india_csv(papers, path)`: writes `INDIA_FIELDS` columns only (extras silently ignored).
`write_nf_csv(papers, path)`: writes `NF_FIELDS` columns.
`write_excluded_csv(excluded, path)`: writes `EXCLUDED_FIELDS` columns.
`format_funnel(funnel) -> list[str]`: pure function, returns lines of PRISMA summary.
`write_funnel(funnel, path)`: calls `format_funnel`, prints to stdout, writes to file.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_classify.py`:

```python
# ── CSV writers ───────────────────────────────────────────────────────────────

def test_write_india_csv(tmp_path):
    papers = [{
        "Corpus ID": "1", "Authors": "Smith J.", "Year": "2021",
        "Title": "Organic farming in Punjab", "DOI": "10.1234/x",
        "Abstract": "Should not appear",  # extra field — must be dropped
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

# ── format_funnel + write_funnel ──────────────────────────────────────────────

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "write_ or format_funnel" -v
```

Expected: all FAILED.

- [ ] **Step 3: Implement CSV writers and funnel functions**

```python
def write_india_csv(papers, path=INDIA_OUT):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDIA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(papers)

def write_nf_csv(papers, path=NF_OUT):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NF_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(papers)

def write_excluded_csv(excluded, path=EXCLUDED_OUT):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXCLUDED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(excluded)

def format_funnel(funnel):
    def val(key):
        return str(funnel.get(key, "N/A"))
    return [
        "PRISMA Funnel Summary",
        "=" * 42,
        f"Total papers read:                {val('total_read')}",
        f"  - Duplicate (Title, Year):      -{val('removed_title_year')}",
        f"  - Duplicate (DOI):              -{val('removed_doi')}",
        f"  - No abstract:                  -{val('removed_no_abstract')}",
        f"Papers to classify (India):       {val('to_classify')}",
        f"  - India: not relevant:          -{val('india_not_relevant')}",
        f"  - India: errors (retry):        {val('india_errors')}",
        f"India-relevant:                   {val('india_relevant')}",
        f"  - Natural farming: not rel:     -{val('nf_not_relevant')}",
        f"  - Natural farming: errors:      {val('nf_errors')}",
        f"Final shortlist:                  {val('nf_relevant')}",
    ]

def write_funnel(funnel, path=FUNNEL_OUT):
    text = "\n".join(format_funnel(funnel))
    print(text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --with pytest --with openai pytest tests/test_classify.py -k "write_ or format_funnel" -v
```

Expected: all PASSED.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
uv run --with pytest --with openai pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add classify_papers.py tests/test_classify.py
git commit -m "Implement and test CSV writers and PRISMA funnel formatter"
```

---

## Task 8: Async machinery — classify, run_step, async_main, main

**Files:**
- Modify: `classify_papers.py`

No unit tests for these (they call the OpenAI API). Correctness is verified by the smoke test in Task 9.

- [ ] **Step 1: Implement `classify`**

Replace the `classify` stub:

```python
async def classify(client, block, system_prompt, semaphore):
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": block},
                    ],
                    response_format=RESPONSE_FORMAT,
                )
            result = json.loads(response.choices[0].message.content)
            return result, response.usage
        except openai.RateLimitError:
            await asyncio.sleep(BASE_DELAY * (2 ** attempt))
        except openai.APIStatusError as e:
            if e.status_code < 500:
                return None, None  # client error, don't retry
            await asyncio.sleep(BASE_DELAY * (2 ** attempt))
        except (openai.APIConnectionError, openai.APITimeoutError):
            await asyncio.sleep(BASE_DELAY * (2 ** attempt))
        except Exception:
            return None, None
    return None, None
```

- [ ] **Step 2: Implement `run_step`**

Replace the `run_step` stub:

```python
async def run_step(client, papers, system_prompt, cache_path, step_label):
    cache = load_cache(cache_path)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    relevant = []
    excluded = []
    errors = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    async def process(paper):
        nonlocal errors
        cid = paper["Corpus ID"]
        if cid in cache:
            result = cache[cid]
        else:
            block = paper_block(paper)
            result, usage = await classify(client, block, system_prompt, semaphore)
            if result is None:
                errors += 1
                return
            if usage:
                total_usage["prompt_tokens"] += usage.prompt_tokens
                total_usage["completion_tokens"] += usage.completion_tokens
            save_to_cache(cache_path, cid, result)

        if result["relevant"]:
            relevant.append({
                **paper,
                f"{step_label}_confidence": result["confidence"],
                f"{step_label}_reason": result["reason"],
            })
        else:
            excluded.append({
                "Corpus ID": cid,
                "Title": paper["Title"],
                "Year": paper.get("Year", ""),
                "Stage": f"{step_label} screen",
                "Reason": result["reason"],
            })

    await asyncio.gather(*[process(p) for p in papers])
    return relevant, excluded, errors, total_usage
```

- [ ] **Step 3: Implement `async_main`**

Replace the `async_main` stub:

```python
async def async_main(smoke=False):
    client = AsyncOpenAI()
    papers, funnel, preprocessing_excluded = load_and_preprocess()

    print(f"Preprocessing: {funnel['total_read']} read → {funnel['to_classify']} to classify "
          f"(−{funnel['removed_title_year']} title-dup, −{funnel['removed_doi']} doi-dup, "
          f"−{funnel['removed_no_abstract']} no-abstract)")

    classify_papers = papers[:3] if smoke else papers
    if smoke:
        print(f"\nSmoke test: classifying first {len(classify_papers)} papers...\n")

    india_papers, india_excluded, india_errors, india_usage = await run_step(
        client, classify_papers, INDIA_PROMPT, INDIA_CACHE, "India"
    )
    funnel.update({
        "india_relevant": len(india_papers),
        "india_not_relevant": len(india_excluded),
        "india_errors": india_errors,
    })

    if smoke:
        print("=== India screen results ===")
        for p in india_papers:
            print(f"  RELEVANT [{p['India_confidence']}]: {p['Title'][:70]}")
            print(f"    → {p['India_reason']}")
        for e in india_excluded:
            print(f"  NOT RELEVANT: {e['Title'][:70]}")
            print(f"    → {e['Reason']}")
        n = len(classify_papers)
        prompt_tok = india_usage["prompt_tokens"]
        completion_tok = india_usage["completion_tokens"]
        total_tok = prompt_tok + completion_tok
        avg = total_tok / max(n, 1)
        projected = int(avg * funnel["to_classify"])
        print(f"\nTokens used ({n} papers): {total_tok} "
              f"(prompt={prompt_tok}, completion={completion_tok})")
        print(f"Average per paper: {avg:.0f} tokens")
        print(f"Projected for full {funnel['to_classify']}-paper India step: ~{projected:,} tokens")
        print("\nIf results look correct, run without --smoke to start the full sweep.")
        return

    nf_papers, nf_excluded, nf_errors, _ = await run_step(
        client, india_papers, NF_PROMPT, NF_CACHE, "NF"
    )
    funnel.update({
        "nf_relevant": len(nf_papers),
        "nf_not_relevant": len(nf_excluded),
        "nf_errors": nf_errors,
    })

    all_excluded = preprocessing_excluded + india_excluded + nf_excluded
    write_india_csv(india_papers)
    write_nf_csv(nf_papers)
    write_excluded_csv(all_excluded)
    write_funnel(funnel)

    print(f"\nWrote {INDIA_OUT} ({len(india_papers)} papers)")
    print(f"Wrote {NF_OUT} ({len(nf_papers)} papers)")
    print(f"Wrote {EXCLUDED_OUT} ({len(all_excluded)} papers)")
    print(f"Wrote {FUNNEL_OUT}")
    if india_errors or nf_errors:
        print(f"\nWARNING: {india_errors + nf_errors} paper(s) errored — rerun to retry them.")
```

- [ ] **Step 4: Verify the full test suite still passes (no regressions from the additions)**

```bash
uv run --with pytest --with openai pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add classify_papers.py
git commit -m "Implement async classify, run_step, and main pipeline"
```

---

## Task 9: Smoke test — verify gpt-5-mini API surface and estimate cost

**Purpose:** Confirm the API call works with gpt-5-mini before committing ~5,469 calls. Check `RESPONSE_FORMAT` is accepted, structured output parses correctly, and get real token counts for cost projection.

- [ ] **Step 1: Confirm `OPENAI_API_KEY` is set**

```bash
echo "Key length: ${#OPENAI_API_KEY}"
```

Expected: a non-zero number (e.g. `Key length: 164`).

- [ ] **Step 2: Run the smoke test**

```bash
uv run classify_papers.py --smoke
```

Expected output (values will differ):

```
Preprocessing: 7364 read → 5469 to classify (−437 title-dup, −39 doi-dup, −1419 no-abstract)

Smoke test: classifying first 3 papers...

=== India screen results ===
  RELEVANT [high]: Agroecology and the social sciences: A half-century...
    → The paper is a systematic review focused on agroecology globally, India relevance unclear.
  NOT RELEVANT: The political economy of the agri-food system in Thailand...
    → The study is set in Thailand, not India.
  ...

Tokens used (3 papers): 1842 (prompt=1680, completion=162)
Average per paper: 614 tokens
Projected for full 5469-paper India step: ~3,357,666 tokens

If results look correct, run without --smoke to start the full sweep.
```

- [ ] **Step 3: Evaluate smoke test output**

Check: Did the structured output parse (no JSON errors)? Do the `relevant`/`reason` judgments look reasonable? Is the token count plausible?

If `RESPONSE_FORMAT` is rejected with an API error: the gpt-5-mini structured-output API may differ from gpt-4o's. Try changing `RESPONSE_FORMAT` to `{"type": "json_object"}` and add an explicit JSON instruction to the end of each prompt:

```
Respond with ONLY a JSON object: {"relevant": <bool>, "confidence": "<high|medium|low>", "reason": "<one sentence>"}
```

Then rerun the smoke test until the output parses correctly.

- [ ] **Step 4: Report projected cost to Pradeep and get approval**

Tell Pradeep the projected token count and estimated cost before running the full sweep. Do NOT proceed to Task 10 without explicit approval.

- [ ] **Step 5: Commit any API compatibility fixes made during smoke test**

```bash
git add classify_papers.py
git commit -m "Adjust API params based on gpt-5-mini smoke test"
```

(Skip if no changes were needed.)

---

## Task 10: Full run

**Prerequisite:** Pradeep has approved the cost estimate from Task 9.

- [ ] **Step 1: Run the full pipeline**

```bash
uv run classify_papers.py
```

This will take several minutes. Progress is visible per API response. Ctrl-C is safe — the caches preserve completed work and the next run resumes from where it stopped.

- [ ] **Step 2: Inspect the outputs**

```bash
# Funnel summary
cat funnel_summary.txt

# Count rows in each output
wc -l india_papers.csv natural_farming_papers.csv excluded_papers.csv

# Spot-check the final shortlist
head -5 natural_farming_papers.csv
```

- [ ] **Step 3: Check for errors**

```bash
grep -c "errors" funnel_summary.txt
```

If the funnel shows errors > 0, rerun `uv run classify_papers.py` to retry only the failed papers (they are not in the cache).

- [ ] **Step 4: Commit outputs**

```bash
git add india_papers.csv natural_farming_papers.csv excluded_papers.csv funnel_summary.txt
git commit -m "Add classifier outputs from full run"
```

---

## Task 11: Rewrite README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite `README.md` from scratch**

```markdown
# nf-review

Two-step LLM classifier for systematic review of academic papers on natural farming in India.

Papers from the input corpus are first screened for substantive India relevance, then for
any sustainable or agroecological farming angle. Each decision includes a model-generated
rationale and confidence level, so the resulting shortlist is auditable.

## Method

**Step 1 — India screen.** Each paper (title + abstract, and keywords when present) is
classified by an LLM as substantively about India or not. "Substantive" means India is a
central focus — the study site, policy context, dataset, or primary case study — not a
passing mention.

**Step 2 — Natural farming screen.** Papers that passed step 1 are classified for any
non-conventional, sustainable, or agroecological farming content: natural farming /
ZBNF / SPNF / APCNF, organic farming, agroecology, permaculture, regenerative agriculture,
conservation agriculture, biodynamic farming, traditional/indigenous practices, or
low-external-input farming.

**Deduplication.** Before classification the corpus is deduplicated: first by normalized
(Title, Year), then by DOI, keeping the richest (most complete) copy each time. Papers
with no abstract are excluded as unclassifiable.

**Resumability.** Each classification step caches results in a JSONL file
(`cache_india.jsonl`, `cache_natural_farming.jsonl`). An interrupted run resumes from
where it stopped at no extra cost.

## Files

| File | Description |
|------|-------------|
| `classify_papers.py` | The classifier script (requires `openai`; run with `uv`). |
| `india_papers.csv` | Papers with substantive India relevance (step 1 survivors). |
| `natural_farming_papers.csv` | **Final shortlist**: India-relevant + natural-farming papers. |
| `excluded_papers.csv` | Every excluded paper with the stage and reason for exclusion. |
| `funnel_summary.txt` | PRISMA-style count from full corpus to final shortlist. |

The raw input corpus (`Total_Paper_List.csv`) is **not** included — it contains
Scopus-sourced abstracts that cannot be redistributed. Place your own copy in the repo
root to reproduce the outputs.

## Input format

`Total_Paper_List.csv` must be a UTF-8 CSV with this header row:

```
Corpus ID,Authors,Year,Title,Abstract,Keywords,Source,DOI,Document Type,Database
```

Export from Scopus or any source with the same fields. Abstracts may contain embedded
newlines (the script uses Python's `csv` module). `[No abstract available]` is treated as
empty; papers with no abstract are excluded in preprocessing.

## Running

Set `OPENAI_API_KEY` in your environment, then:

```sh
# Smoke test first: 3 papers, prints results + cost estimate
uv run classify_papers.py --smoke

# Full run (after approving the cost estimate)
uv run classify_papers.py
```

`uv` provisions the interpreter and `openai` dependency from the script's PEP 723 header.
```

- [ ] **Step 2: Commit the README**

```bash
git add README.md
git commit -m "Rewrite README for LLM classifier pipeline"
```

---

## Task 12: Merge

- [ ] **Step 1: Verify all tests pass on the branch**

```bash
uv run --with pytest --with openai pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 2: Merge to main**

```bash
git checkout main
git merge llm-classifier
```

- [ ] **Step 3: Confirm clean state**

```bash
git log --oneline -10
git status
```

Expected: clean working tree, new commits visible on `main`.
