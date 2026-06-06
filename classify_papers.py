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
    return " ".join((title or "").strip().lower().split())

def has_abstract(row):
    a = (row.get("Abstract") or "").strip()
    return a != "" and a != NO_ABSTRACT

def text_completeness(row):
    abstract = "" if (row.get("Abstract") or "").strip() == NO_ABSTRACT else (row.get("Abstract") or "")
    return len(abstract) + len(row.get("Keywords") or "")

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

    # Build excluded rows — attribute each dropped paper to its earliest stage
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

# ── TEXT ASSEMBLY ─────────────────────────────────────────────────────────────

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

# ── CACHE ─────────────────────────────────────────────────────────────────────

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

# ── CSV WRITERS + FUNNEL ──────────────────────────────────────────────────────

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
