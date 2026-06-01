# nf-review

Relevance scoring of academic papers for a systematic review. Each paper is matched against a
controlled vocabulary of terms grouped into four categories and assigned a weighted score, so
the most relevant papers rise to the top.

## Files

| File | Description |
|------|-------------|
| `score_papers.py` | The scoring script (stdlib only; run with `uv`). |
| `Terms.csv` | Controlled vocabulary: `Term, Category` (categories: DSD, LSD, PSD, MSD). |
| `paper_scores_ranked.csv` | All deduplicated papers, ranked by score. |
| `top_100.csv` | The top 100 papers (change `TOP_N` in the script for a different cut). |
| `sample_papers.csv` | A 50-row sanitized sample of the input corpus (abstracts removed) showing the expected format. |
| `make_sample.py` | How `sample_papers.csv` was derived from the full corpus. |

The raw input corpus (`Total_Paper_List.csv`) is **not** included: it contains
Scopus-sourced abstracts that cannot be redistributed publicly. Place your own copy in the
repo root to reproduce the outputs.

## Data

To reproduce the score CSVs you supply your own `Total_Paper_List.csv` in the repo root (it is
gitignored, so it will never be committed). Export it from Scopus (or any source with the same
fields) as a UTF-8 CSV with this header row:

```
Corpus ID,Authors,Year,Title,Abstract,Keywords,Source,DOI,Document Type,Database
```

Notes:

- Only `Title`, `Abstract`, `Keywords`, `Year`, and `DOI` affect scoring and dedup; the other
  columns are carried through to the output for reference.
- Abstracts may contain embedded newlines — keep them properly quoted so a CSV parser reads
  one record per paper (the script uses Python's `csv` module, which handles this).
- An abstract of `[No abstract available]` is treated as empty.
- Missing `Abstract`/`Keywords` are fine; the paper is still scored on whatever text is present.

Then run the scorer (see **Running** below) to regenerate `paper_scores_ranked.csv` and
`top_100.csv`.

`sample_papers.csv` is a 50-row sanitized sample (real bibliographic metadata; abstracts
removed) that illustrates this exact format — useful as a template when preparing your own
corpus.

## Method

**Deduplication.** Rows are first deduplicated by normalized `(Title, Year)`, then collapsed
again by `DOI` (keeping the row with the most complete abstract + keywords).

**Matching.** For each paper the searchable unit is `Title + Abstract + Keywords`, lowercased.
Matching is case-insensitive:

- non-wildcard terms match as whole words (`biomass` → `\bbiomass\b`);
- a trailing `*` is a truncation wildcard — the stem may continue within the word
  (`climat*` → `\bclimat\w*`, matching *climate*, *climatic*, …);
- multi-word terms match as a phrase with flexible whitespace.

For each category a paper gets two metrics:

- **variety** — the number of *distinct* terms matched;
- **frequency** — the *total count* of all matches across all terms (raw occurrences).

**Score.**

```
Score = DSD_variety×7 + DSD_frequency×3
      + LSD_variety×4 + LSD_frequency×1
      + PSD_variety×1 + PSD_frequency×0.25
      + MSD_variety×0.5 + MSD_frequency×0.1
```

## Output columns

`Rank, Corpus ID, Authors, Year, Title, DOI, DSD_variety, DSD_frequency, LSD_variety,
LSD_frequency, PSD_variety, PSD_frequency, MSD_variety, MSD_frequency, Score, Matched_Terms`

`Matched_Terms` lists the distinct terms found, grouped by category with per-term counts, e.g.
`DSD: caste*(4); gender*(2) | LSD: agraria(1)`.

## Running

```sh
uv run score_papers.py
```

`uv` provisions the interpreter from the script's PEP 723 header; no manual virtualenv needed.
