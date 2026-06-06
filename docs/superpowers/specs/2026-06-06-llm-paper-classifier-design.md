# LLM Paper Classifier — Design

**Date:** 2026-06-06
**Status:** Approved (design); pending spec review
**Author:** Pradeep & Claude

## Problem

The keyword/vocabulary scoring approach (`score_papers.py` + `Terms.csv`) did not
discriminate relevant papers well. A weighted count of controlled-vocabulary term
matches cannot tell whether a paper is *centrally about* India, or whether it actually
concerns an alternative farming approach — it only counts surface term occurrences.

We are replacing it wholesale with a two-step LLM classifier that makes a judgment per
paper, with a recorded rationale, so the resulting corpus is defensible for a systematic
review.

## Goal

From a corpus of ~7,364 papers, produce a final shortlist of papers that are **both**:

1. **Substantively about India** — India (or a specific Indian state/region/city) is a
   *central* focus, not a passing mention; and
2. **About natural farming, broadly** — any non-conventional / sustainable /
   agroecological farming approach.

Each included or excluded paper carries a short machine-generated rationale and a
confidence level, so decisions are auditable. The pipeline also produces a PRISMA-style
funnel accounting — the count of papers from full corpus to final shortlist, with the
number rejected and the reason at every stage.

## Non-goals (YAGNI)

- No numeric relevance score or ranking — the output is a yes/no shortlist per step.
- No re-export of a sanitized public sample corpus (the new output CSVs contain only
  bibliographic metadata, no abstracts, so they are already publishable).
- No web UI, no database — flat CSV in, flat CSV out.
- No batching of multiple papers per API call (rejected for auditability/reliability).

## Definitions (decided with Pradeep)

**India relevance (step 1) — "substantive only":** `relevant = true` only when India, or a
specific Indian state/region/city, is a *central* subject of the paper (study site,
policy context, studied population, dataset, or case study). A paper that merely mentions
India in passing, or lists it among many countries without India-specific content, is
`false`.

**Natural farming (step 2) — "broad":** `relevant = true` for *any* non-conventional,
sustainable, or agroecological farming approach, including but not limited to: natural
farming / Zero Budget Natural Farming (ZBNF) / Subhash Palekar Natural Farming (SPNF) /
Andhra Pradesh Community-managed Natural Farming (APCNF), organic farming, agroecology,
permaculture, regenerative agriculture, conservation agriculture, biodynamic farming,
traditional / indigenous farming practices, and low-external-input or chemical-free
farming. Purely conventional/industrial agriculture with no alternative-farming angle is
`false`.

## Data flow

```
Total_Paper_List.csv  (7,364 rows, gitignored, user-supplied)
        │
        ▼  dedup (Title,Year) keep richest      − 437
        ▼  dedup (DOI) keep richest             −  39
        ▼  remove papers with no abstract       −1419
   5,469 papers to classify
        │
        ▼  STEP 1 — India? (gpt-5-mini, over the 5,469)
   india_papers.csv          (survivors + India_confidence + India_reason)
        │
        ▼  STEP 2 — Natural farming? (gpt-5-mini, over step-1 survivors only)
   natural_farming_papers.csv   ← FINAL deliverable
```

(Stage counts above are from the current `Total_Paper_List.csv`; they will vary with a
different corpus. The LLM-step exclusion counts are not yet known — they come from the
run.)

Step 2 runs only over step-1 survivors, so its prompt assumes India relevance and judges
only the farming dimension.

## Inputs & environment

- `Total_Paper_List.csv` — header:
  `Corpus ID,Authors,Year,Title,Abstract,Keywords,Source,DOI,Document Type,Database`.
  Gitignored; user supplies their own copy. `[No abstract available]` is treated as empty.
- `OPENAI_API_KEY` — read from the environment.
- Run via `uv run classify_papers.py` (PEP 723 header declares `openai` dependency).

## Preprocessing: dedup + abstract filter

Method-agnostic preprocessing, re-implemented cleanly (not carried over from the old
script). Three stages, in this order:

1. **Dedup by `(Title, Year)`** — normalized (lowercased, whitespace-collapsed) title.
   Among rows sharing a key, keep the one with the most complete searchable text
   (`len(abstract) + len(keywords)`, `[No abstract available]` counted as empty).
2. **Dedup by `DOI`** — lowercased DOI; again keep the most complete row. Rows with no
   DOI are all kept.
3. **Remove papers with no abstract** — drop any remaining row whose abstract is empty or
   `[No abstract available]`. These cannot be judged on more than a title, so they are
   excluded rather than classified on a title alone.

**Why dedup before the abstract filter.** Both dedup stages keep the *richest* copy, so a
paper that exists both with and without an abstract keeps its abstract-bearing copy. Doing
dedup first means the "no abstract" exclusion counts only papers for which *no* copy has
an abstract anywhere — an honest figure — while a duplicate that merely happened to lack
an abstract is correctly attributed to "duplicate." Verified: the final set is identical
either order (5,469 papers); only the attribution of a removed copy differs.

All preprocessing happens once, before any API calls, so removed papers are never
classified or listed.

## Paper text assembly

For each paper the model receives a labeled block built from `Title`, `Year`, `Abstract`,
and `Keywords`. Because papers with no abstract are removed in preprocessing (see below),
the model always has **at least a title and an abstract** to judge. Keywords are often
absent (5,996 of the full corpus lack them) and are simply omitted from the block when
empty. The `confidence` field still flags cases the model finds thin or ambiguous.

## Classification core

A single function reused for both steps; only the system prompt differs:

```
classify(paper_block: str, system_prompt: str, model: str) -> Result
```

**Structured output** — the call uses OpenAI structured outputs (JSON schema via
`response_format`) so the model must return valid JSON matching:

```json
{
  "type": "object",
  "properties": {
    "relevant":   {"type": "boolean"},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "reason":     {"type": "string"}
  },
  "required": ["relevant", "confidence", "reason"],
  "additionalProperties": false
}
```

The `reason` prompt instructs the model to describe its judgment generically and **not
quote the abstract**, so no licensed text lands in committed output.

**Error handling.** Retries with exponential backoff on rate-limit (429), 5xx, and
timeout errors. After exhausting retries, the paper is recorded with an `error` status —
**never silently treated as not-relevant.** Errors are counted in the run summary and
retried on the next run (they are not written to the cache as decisions).

## Prompts

**Step 1 — India.** System prompt states the "substantive only" criterion above with 2–3
worked examples (central focus → yes; multi-country passing mention → no; Indian state
case study → yes). Asks for `relevant`, `confidence`, `reason`.

**Step 2 — Natural farming.** System prompt states the "broad" criterion above with
examples spanning ZBNF/organic/agroecology/traditional practices → yes; purely
conventional/industrial agriculture → no. Inputs are already India-filtered, so the prompt
does not re-check India.

## Caching & resumability

One JSONL cache per step: `cache_india.jsonl`, `cache_natural_farming.jsonl`. Each line:
`{"corpus_id": ..., "relevant": ..., "confidence": ..., "reason": ...}`. Before each call
the cache is consulted and hits are skipped. A multi-thousand-call run (~5,469 for step 1)
can be interrupted (crash, rate limit, Ctrl-C) and resumed at no extra cost — only
unclassified or previously-errored papers re-fire. `error` outcomes are **not** cached, so they retry next run.

Caches are **gitignored** as transient build artifacts (they are large and a `reason` may
paraphrase abstract content).

## Concurrency

`asyncio` + `AsyncOpenAI` with a semaphore bounding in-flight requests (config constant,
default 10). I/O-bound fan-out with clean cancellation. Concurrency is tunable down if we
hit rate limits during the smoke test.

## Outputs

| File | Columns / contents |
|------|---------|
| `india_papers.csv` | `Corpus ID, Authors, Year, Title, DOI, India_confidence, India_reason` |
| `natural_farming_papers.csv` (**final**) | `Corpus ID, Authors, Year, Title, DOI, India_confidence, India_reason, NF_confidence, NF_reason` |
| `excluded_papers.csv` | `Corpus ID, Title, Year, Stage, Reason` — every rejected paper, all stages |
| `funnel_summary.txt` | PRISMA-style counts: full → final, removed-with-reason per stage |

The two shortlist CSVs hold only the `relevant = true` survivors of their step:
`india_papers.csv` is the India-relevant set, and `natural_farming_papers.csv` is the
both-criteria shortlist (its rows are India-relevant by construction *and*
natural-farming-relevant). All CSVs contain only bibliographic metadata (no abstracts) and
are safe to commit. The complete per-paper decision record (including model reasons) also
lives in the JSONL caches.

## Funnel accounting (PRISMA-style)

The pipeline tracks every paper from the full corpus to the final shortlist, recording how
many were rejected and why at each stage:

1. **Total read** from `Total_Paper_List.csv`
2. **− duplicate (Title, Year)**
3. **− duplicate (DOI)**
4. **− no abstract**
5. **= papers classified (India step)**
6. **− India screen: not relevant** (and **error** count, if any)
7. **= India-relevant (→ `india_papers.csv`)**
8. **− natural-farming screen: not relevant** (and **error** count, if any)
9. **= final shortlist (→ `natural_farming_papers.csv`)**

These counts are printed to stdout at the end of the run **and** persisted to
`funnel_summary.txt`. Per-paper attribution for every rejected paper — Corpus ID, the
stage that rejected it, and the reason (categorical for preprocessing; the model's reason
for the screens) — is written to `excluded_papers.csv`, giving a complete audit trail in
one place. `error` outcomes are reported in the funnel but are not exclusions (they retry
on the next run).

## Smoke test & cost gating

Before the full sweep:

1. Run the classifier on **3 papers** to empirically confirm gpt-5-mini's API surface —
   whether it takes `max_completion_tokens` vs `max_tokens`, whether `temperature` is
   fixed, whether `reasoning_effort` applies, and that structured outputs work.
2. Report **real token usage** from those calls and extrapolate the full-corpus cost.
   Pradeep approves the full run with actual numbers before the ~5,469 step-1 calls fire.

## Config constants (top of script)

`MODEL = "gpt-5-mini"`, file paths, `CONCURRENCY = 10`, retry/backoff parameters, the two
system prompts, and the structured-output schema. The model id and call params are
constants so they can be bumped without touching logic.

## Code structure

A single script `classify_papers.py` with a PEP 723 header (`requires-python`, deps
`["openai"]`), organized in clear sections: config/prompts → corpus IO & dedup → text
assembly → cache → async classify (with retry) → step drivers → CSV writers → `main()`.
Single script matches the old repo's style; it will be split only if it outgrows one file.

## Repository changes

**Delete** (old keyword-method artifacts): `score_papers.py`, `make_sample.py`,
`Terms.csv`, `paper_scores_ranked.csv`, `top_100.csv`, `sample_papers.csv`.

**Rewrite from scratch:** `README.md` — document the new two-step LLM pipeline, input
format, how to run, outputs, and the `OPENAI_API_KEY` requirement.

**Update:** `.gitignore` — keep `Total_Paper_List.csv` ignored; add `cache_*.jsonl`.

**Keep:** `LICENSE`, `Total_Paper_List.csv` (gitignored input).

## Testing

- Unit tests on **pure logic only** with small real fixtures: dedup, no-abstract filter,
  funnel counting, paper-text assembly, cache read/write round-trip, CSV writing. No
  mocked-OpenAI tests (they would only test the mock).
- The live API path is verified by the 3-paper smoke test, not a mock.
- A full end-to-end over 7k papers is gated on the cost approval above.

## Risks & uncertainties

- **gpt-5-mini API surface** is past the assistant's Aug 2025 knowledge cutoff. The exact
  param names and structured-output mechanism are verified empirically in the smoke test
  before committing to the full run, rather than assumed.
- **Papers with no abstract are excluded** in preprocessing (1,419 in the current corpus).
  This trades recall for judgment quality — a title alone is too thin to classify
  reliably. They are recorded in `excluded_papers.csv` so the loss is visible and
  reviewable, not silent.
- **Abstract-only papers** (abstract present but no keywords, ~5,996 in the full corpus)
  are still classified on title + abstract; `confidence` flags any the model finds thin.
