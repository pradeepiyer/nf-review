# nf-review

Three-step LLM classifier for systematic review of academic papers on natural farming in India.

Papers from the input corpus are screened in sequence: first for substantive India relevance,
then for any sustainable or agroecological farming angle, and optionally for social dimensions
of farming. Each decision includes a model-generated rationale and confidence level, so the
resulting shortlist is auditable.

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

**Step 3 — Social dimensions screen (optional).** Papers that passed step 2 are classified
for substantive social dimensions: gender, age, caste, poverty, livelihoods, land ownership,
mobility and migrant status, political affiliations, culture and traditions, community
structures, or socio-economic conditions. Run separately with `--social`.

**Resumability.** Each classification step caches results in a JSONL file
(`cache_india.jsonl`, `cache_natural_farming.jsonl`, `cache_social.jsonl`). An interrupted
run resumes from where it stopped at no extra cost.

## Files

| File | Description |
|------|-------------|
| `classify_papers.py` | The classifier script (requires `openai`; run with `uv`). |
| `india_papers.csv` | Papers with substantive India relevance (step 1 survivors). |
| `natural_farming_papers.csv` | India-relevant + natural-farming papers (step 2 survivors). |
| `social_dimensions_papers.csv` | Natural farming papers with a social angle (step 3 survivors). |
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

# Full run (steps 1 + 2: India screen → natural farming screen)
uv run classify_papers.py

# Step 3: social dimensions screen (runs on natural_farming_papers.csv)
uv run classify_papers.py --social
```

`uv` provisions the interpreter and `openai` dependency from the script's PEP 723 header.
