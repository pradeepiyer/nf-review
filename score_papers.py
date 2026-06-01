# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Score papers in Total_Paper_List.csv against the controlled vocabulary in Terms.csv.

Each paper is scored on how many distinct terms it matches (variety) and how often
(frequency) within four categories, weighted to favour DSD over LSD over PSD over MSD.
Outputs a full ranked CSV plus a top-N slice, both tuned for finding the most relevant papers.
"""

import csv
import re
import collections

TERMS_FILE = "Terms.csv"
PAPERS_FILE = "Total_Paper_List.csv"
RANKED_OUT = "paper_scores_ranked.csv"
TOP_N = 100
TOP_N_OUT = f"top_{TOP_N}.csv"

NO_ABSTRACT = "[No abstract available]"

CATEGORIES = ["DSD", "LSD", "PSD", "MSD"]

# (variety weight, frequency weight) per category
WEIGHTS = {
    "DSD": (7, 3),
    "LSD": (4, 1),
    "PSD": (1, 0.25),
    "MSD": (0.5, 0.1),
}

csv.field_size_limit(10 ** 8)


def compile_term(term):
    """Build a case-insensitive whole-word regex for a vocabulary term.

    A trailing '*' is a truncation wildcard: the stem may continue within the word
    (climat* -> climate, climatic). Internal spaces match flexible whitespace so
    multi-word phrases match across line breaks.
    """
    wildcard = term.endswith("*")
    stem = term[:-1] if wildcard else term
    parts = [re.escape(p) for p in stem.split()]
    pattern = r"\s+".join(parts)
    pattern = r"\b" + pattern
    pattern += r"\w*" if wildcard else r"\b"
    return re.compile(pattern, re.IGNORECASE)


def load_terms():
    terms = []
    with open(TERMS_FILE, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            term = row["Term"].strip()
            category = row["Category"].strip()
            if term and category:
                terms.append((compile_term(term), term, category))
    return terms


def norm_title(title):
    return " ".join((title or "").strip().lower().split())


def text_completeness(row):
    """Length of the searchable text, used to pick the richest row among DOI duplicates."""
    abstract = "" if row["Abstract"].strip() == NO_ABSTRACT else row["Abstract"]
    return len(abstract) + len(row["Keywords"] or "")


def load_and_dedup():
    with open(PAPERS_FILE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)

    seen_title_year = set()
    kept = []
    removed_title_year = 0
    for row in rows:
        key = (norm_title(row["Title"]), (row["Year"] or "").strip())
        if key in seen_title_year:
            removed_title_year += 1
            continue
        seen_title_year.add(key)
        kept.append(row)

    by_doi = {}
    no_doi = []
    removed_doi = 0
    for row in kept:
        doi = (row["DOI"] or "").strip().lower()
        if not doi:
            no_doi.append(row)
            continue
        if doi in by_doi:
            removed_doi += 1
            if text_completeness(row) > text_completeness(by_doi[doi]):
                by_doi[doi] = row
        else:
            by_doi[doi] = row

    deduped = list(by_doi.values()) + no_doi
    return deduped, total, removed_title_year, removed_doi


def paper_text(row):
    abstract = "" if row["Abstract"].strip() == NO_ABSTRACT else row["Abstract"]
    return f"{row['Title']} {abstract} {row['Keywords']}"


def score_paper(text, terms):
    variety = collections.Counter()
    frequency = collections.Counter()
    matched = collections.defaultdict(list)  # category -> [(term, count), ...]
    for regex, term, category in terms:
        hits = regex.findall(text)
        if hits:
            variety[category] += 1
            frequency[category] += len(hits)
            matched[category].append((term, len(hits)))

    score = 0.0
    for category in CATEGORIES:
        v_weight, f_weight = WEIGHTS[category]
        score += variety[category] * v_weight + frequency[category] * f_weight

    return variety, frequency, score, matched


def format_matched(matched):
    blocks = []
    for category in CATEGORIES:
        if matched.get(category):
            items = "; ".join(f"{term}({count})" for term, count in matched[category])
            blocks.append(f"{category}: {items}")
    return " | ".join(blocks)


def build_record(row, variety, frequency, score, matched):
    record = {
        "Corpus ID": row["Corpus ID"],
        "Authors": row["Authors"],
        "Year": row["Year"],
        "Title": row["Title"],
        "DOI": row["DOI"],
    }
    for category in CATEGORIES:
        record[f"{category}_variety"] = variety[category]
        record[f"{category}_frequency"] = frequency[category]
    record["Score"] = round(score, 4)
    record["Matched_Terms"] = format_matched(matched)
    return record


FIELDNAMES = (
    ["Rank", "Corpus ID", "Authors", "Year", "Title", "DOI"]
    + [f"{c}_{m}" for c in CATEGORIES for m in ("variety", "frequency")]
    + ["Score", "Matched_Terms"]
)


def write_csv(path, records):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def median(values):
    if not values:
        return 0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def main():
    terms = load_terms()
    deduped, total, removed_title_year, removed_doi = load_and_dedup()

    records = []
    for row in deduped:
        variety, frequency, score, matched = score_paper(paper_text(row), terms)
        records.append(build_record(row, variety, frequency, score, matched))

    records.sort(key=lambda r: r["Score"], reverse=True)
    for rank, record in enumerate(records, start=1):
        record["Rank"] = rank

    write_csv(RANKED_OUT, records)
    write_csv(TOP_N_OUT, records[:TOP_N])

    scores = [r["Score"] for r in records]
    zero = sum(1 for s in scores if s == 0)

    print(f"Terms loaded: {len(terms)}")
    print(f"Papers read:           {total}")
    print(f"Removed (title+year):  {removed_title_year}")
    print(f"Removed (DOI):         {removed_doi}")
    print(f"Papers scored (out):   {len(records)}")
    print()
    print(f"Score min/median/max:  {min(scores)} / {median(scores)} / {max(scores)}")
    print(f"Zero-score papers:     {zero}")
    print()
    print(f"Wrote {RANKED_OUT} ({len(records)} rows) and {TOP_N_OUT} ({min(TOP_N, len(records))} rows)")
    print()
    print("Top 20:")
    for r in records[:20]:
        title = r["Title"][:70]
        print(f"  #{r['Rank']:<3} {r['Score']:>8}  {r['Year']}  {title}")


if __name__ == "__main__":
    main()
