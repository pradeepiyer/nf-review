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
SOCIAL_CACHE = "cache_social.jsonl"
INDIA_OUT = "india_papers.csv"
NF_OUT = "natural_farming_papers.csv"
SOCIAL_OUT = "social_dimensions_papers.csv"
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

SOCIAL_PROMPT = """\
You are classifying academic papers about natural farming in India for social dimensions. These papers have already been confirmed to be about India and to involve alternative or sustainable farming.

Your task: Determine whether this paper substantively studies one or more SOCIAL DIMENSIONS of farming. Include:
- Gender roles or women's participation in farming decisions or labour
- Age or generational factors among farmers
- Caste dynamics in agricultural access, participation, or outcomes
- Poverty, livelihoods, or income effects on farming households
- Socio-economic conditions or class distinctions among farmers
- Land ownership, tenure security, or land access
- Mobility, migrant workers, or seasonal labour in farming
- Political affiliations, farmer movements, or policy participation
- Culture, traditions, rituals, or indigenous knowledge tied to farming practices
- Community or social structures (collectives, cooperatives, self-help groups)
- Geographic or location-based social inequalities (urban–rural, region, remoteness)

Papers focused PURELY on agronomic or biophysical outcomes (yields, soil chemistry, pest biology, crop genetics) with no social angle are NOT relevant.

Examples:
- "Women's role in ZBNF decision-making in Andhra Pradesh households" → relevant (gender)
- "Caste-based access to natural farming training programmes in Maharashtra" → relevant (caste, social structures)
- "Livelihoods of smallholder farmers transitioning away from chemical inputs" → relevant (poverty, socio-economic)
- "Land tenure and adoption of organic farming in tribal areas of Jharkhand" → relevant (land ownership)
- "Seasonal migrant labour and natural farming calendar conflicts in Punjab" → relevant (mobility, migrant status)
- "Gandhian culture and traditional seed-saving practices in rural Rajasthan" → relevant (culture, traditions)
- "Effect of vermicompost on groundnut yield and phosphorus uptake in West Bengal" → not relevant (purely agronomic)
- "Soil carbon sequestration under zero-tillage in Punjab Vertisols" → not relevant (purely biophysical)

Respond with a JSON object:
- relevant: true if any social dimension is substantively studied, false if purely agronomic or biophysical
- confidence: "high" if clear, "medium" if uncertain, "low" if focus is hard to determine
- reason: one sentence describing your judgment — do NOT quote the abstract
"""

# ── OUTPUT SCHEMAS ────────────────────────────────────────────────────────────

INDIA_FIELDS = ["Corpus ID", "Authors", "Year", "Title", "DOI", "India_confidence", "India_reason"]
NF_FIELDS = ["Corpus ID", "Authors", "Year", "Title", "DOI", "India_confidence", "India_reason", "NF_confidence", "NF_reason"]
SOCIAL_FIELDS = ["Corpus ID", "Authors", "Year", "Title", "DOI", "India_confidence", "India_reason", "NF_confidence", "NF_reason", "Social_confidence", "Social_reason"]
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

def write_social_csv(papers, path=SOCIAL_OUT):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOCIAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(papers)

def write_excluded_csv(excluded, path=EXCLUDED_OUT):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXCLUDED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(excluded)

def count_csv_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)

def count_cache_entries(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def build_funnel_from_files():
    """Reconstruct funnel counts from on-disk files; used by --social to extend the main funnel."""
    _, funnel, _ = load_and_preprocess()
    india_cached = count_cache_entries(INDIA_CACHE)
    india_relevant = count_csv_rows(INDIA_OUT)
    nf_cached = count_cache_entries(NF_CACHE)
    nf_relevant = count_csv_rows(NF_OUT)
    funnel.update({
        "india_relevant": india_relevant,
        "india_not_relevant": india_cached - india_relevant,
        "india_errors": funnel["to_classify"] - india_cached,
        "nf_relevant": nf_relevant,
        "nf_not_relevant": nf_cached - nf_relevant,
        "nf_errors": india_relevant - nf_cached,
    })
    return funnel

def format_funnel(funnel):
    def val(key):
        return str(funnel.get(key, "N/A"))
    lines = [
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
        f"Natural farming shortlist:        {val('nf_relevant')}",
    ]
    if "social_relevant" in funnel:
        lines += [
            f"  - Social dimensions: not rel:   -{val('social_not_relevant')}",
            f"  - Social dimensions: errors:    {val('social_errors')}",
            f"Social dimensions shortlist:      {val('social_relevant')}",
        ]
    return lines

def write_funnel(funnel, path=FUNNEL_OUT):
    text = "\n".join(format_funnel(funnel))
    print(text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

# ── ASYNC CLASSIFIER ──────────────────────────────────────────────────────────

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

# ── MAIN ──────────────────────────────────────────────────────────────────────

async def async_main(smoke=False, social=False):
    client = AsyncOpenAI()

    if social:
        if not os.path.exists(NF_OUT):
            print(f"ERROR: {NF_OUT} not found. Run the full pipeline first.")
            return
        with open(NF_OUT, encoding="utf-8", newline="") as f:
            nf_papers = list(csv.DictReader(f))
        print(f"Social screen: classifying {len(nf_papers)} natural farming papers...")
        social_papers, social_excluded, social_errors, _ = await run_step(
            client, nf_papers, SOCIAL_PROMPT, SOCIAL_CACHE, "Social"
        )
        funnel = build_funnel_from_files()
        funnel.update({
            "social_relevant": len(social_papers),
            "social_not_relevant": len(social_excluded),
            "social_errors": social_errors,
        })
        existing_excluded = []
        if os.path.exists(EXCLUDED_OUT):
            with open(EXCLUDED_OUT, encoding="utf-8", newline="") as f:
                existing_excluded = list(csv.DictReader(f))
        write_social_csv(social_papers)
        write_excluded_csv(existing_excluded + social_excluded)
        write_funnel(funnel)
        print(f"\nWrote {SOCIAL_OUT} ({len(social_papers)} papers)")
        print(f"Wrote {EXCLUDED_OUT} ({len(existing_excluded) + len(social_excluded)} papers total)")
        print(f"Wrote {FUNNEL_OUT}")
        if social_errors:
            print(f"\nWARNING: {social_errors} paper(s) errored — rerun --social to retry.")
        return

    papers, funnel, preprocessing_excluded = load_and_preprocess()

    print(f"Preprocessing: {funnel['total_read']} read → {funnel['to_classify']} to classify "
          f"(−{funnel['removed_title_year']} title-dup, −{funnel['removed_doi']} doi-dup, "
          f"−{funnel['removed_no_abstract']} no-abstract)")

    classify_papers_list = papers[:3] if smoke else papers
    if smoke:
        print(f"\nSmoke test: classifying first {len(classify_papers_list)} papers...\n")

    india_papers, india_excluded, india_errors, india_usage = await run_step(
        client, classify_papers_list, INDIA_PROMPT, INDIA_CACHE, "India"
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
        n = len(classify_papers_list)
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

def main():
    parser = argparse.ArgumentParser(description="Two-step LLM classifier for academic papers.")
    parser.add_argument("--smoke", action="store_true", help="Run on first 3 papers only to verify API and estimate cost.")
    parser.add_argument("--social", action="store_true", help="Run Step 3 (social dimensions screen) on natural_farming_papers.csv.")
    args = parser.parse_args()
    asyncio.run(async_main(smoke=args.smoke, social=args.social))

if __name__ == "__main__":
    main()
