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
confidence level, so decisions are auditable.

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
        ▼  dedup (Title,Year) then DOI
   deduped corpus
        │
        ▼  STEP 1 — India? (gpt-5-mini, over full deduped corpus)
   india_papers.csv          (survivors + India_confidence + India_reason)
        │
        ▼  STEP 2 — Natural farming? (gpt-5-mini, over step-1 survivors only)
   natural_farming_papers.csv   ← FINAL deliverable
```

Step 2 runs only over step-1 survivors, so its prompt assumes India relevance and judges
only the farming dimension.

## Inputs & environment

- `Total_Paper_List.csv` — header:
  `Corpus ID,Authors,Year,Title,Abstract,Keywords,Source,DOI,Document Type,Database`.
  Gitignored; user supplies their own copy. `[No abstract available]` is treated as empty.
- `OPENAI_API_KEY` — read from the environment.
- Run via `uv run classify_papers.py` (PEP 723 header declares `openai` dependency).

## Deduplication

Method-agnostic preprocessing, re-implemented cleanly (not carried over from the old
script):

1. Drop exact duplicates by normalized `(Title, Year)` — lowercased, whitespace-collapsed
   title.
2. Collapse remaining rows by lowercased `DOI`, keeping the row with the most complete
   searchable text (`len(abstract) + len(keywords)`, with `[No abstract available]`
   counted as empty). Rows with no DOI are all kept.

Dedup happens once, before any API calls, so duplicates are never classified or listed
twice.

## Paper text assembly

For each paper the model receives a labeled block built from `Title`, `Year`, `Abstract`
(empty if `[No abstract available]`), and `Keywords`. Corpus reality: 5,996 papers have no
keywords and 1,437 have no abstract, so many papers are judged on **title alone**. The
prompt instructs the model to judge from whatever text is present; the `confidence` field
flags thin/title-only cases.

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
the cache is consulted and hits are skipped. A 7k-call run can be interrupted (crash, rate
limit, Ctrl-C) and resumed at no extra cost — only unclassified or previously-errored
papers re-fire. `error` outcomes are **not** cached, so they retry next run.

Caches are **gitignored** as transient build artifacts (they are large and a `reason` may
paraphrase abstract content).

## Concurrency

`asyncio` + `AsyncOpenAI` with a semaphore bounding in-flight requests (config constant,
default 10). I/O-bound fan-out with clean cancellation. Concurrency is tunable down if we
hit rate limits during the smoke test.

## Outputs

| File | Columns |
|------|---------|
| `india_papers.csv` | `Corpus ID, Authors, Year, Title, DOI, India_confidence, India_reason` |
| `natural_farming_papers.csv` (**final**) | `Corpus ID, Authors, Year, Title, DOI, India_confidence, India_reason, NF_confidence, NF_reason` |

Each CSV holds only the `relevant = true` survivors of its step: `india_papers.csv` is the
India-relevant set, and `natural_farming_papers.csv` is the both-criteria shortlist (its
rows are India-relevant by construction *and* natural-farming-relevant). The complete
decision record for every paper, including the excluded ones and their reasons, lives in
the JSONL caches. Both CSVs contain only bibliographic metadata (no abstracts) and are
safe to commit.

## Smoke test & cost gating

Before the full sweep:

1. Run the classifier on **3 papers** to empirically confirm gpt-5-mini's API surface —
   whether it takes `max_completion_tokens` vs `max_tokens`, whether `temperature` is
   fixed, whether `reasoning_effort` applies, and that structured outputs work.
2. Report **real token usage** from those calls and extrapolate the full-corpus cost.
   Pradeep approves the full run with actual numbers before ~7k calls fire.

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

- Unit tests on **pure logic only** with small real fixtures: dedup, paper-text assembly,
  cache read/write round-trip, CSV writing. No mocked-OpenAI tests (they would only test
  the mock).
- The live API path is verified by the 3-paper smoke test, not a mock.
- A full end-to-end over 7k papers is gated on the cost approval above.

## Risks & uncertainties

- **gpt-5-mini API surface** is past the assistant's Aug 2025 knowledge cutoff. The exact
  param names and structured-output mechanism are verified empirically in the smoke test
  before committing to the full run, rather than assumed.
- **Title-only papers** (no abstract/keywords) give the model little to judge; `confidence`
  surfaces these so Pradeep can review low-confidence inclusions/exclusions if desired.
