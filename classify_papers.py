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
