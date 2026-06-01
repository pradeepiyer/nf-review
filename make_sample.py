# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Derive a sanitized sample of the paper corpus for public sharing.

Selects 50 papers (the top 25 by Score plus 25 evenly spaced across the rest of the score
distribution) and writes sample_papers.csv with the same columns as Total_Paper_List.csv but
with every Abstract replaced by "[No abstract available]" — the abstracts are the only
Scopus-licensed content, so removing them makes the bibliographic metadata safe to publish.
"""

import csv

CORPUS_FILE = "Total_Paper_List.csv"
RANKED_FILE = "paper_scores_ranked.csv"
SAMPLE_FILE = "sample_papers.csv"
NO_ABSTRACT = "[No abstract available]"

N_TOP = 25
N_SPREAD = 25

csv.field_size_limit(10 ** 8)


def select_corpus_ids():
    """Top N_TOP by score, then N_SPREAD evenly spaced across the remainder. Deterministic."""
    with open(RANKED_FILE, encoding="utf-8-sig", newline="") as f:
        ranked = [row["Corpus ID"] for row in csv.DictReader(f)]

    top = ranked[:N_TOP]
    rest = ranked[N_TOP:]
    spread_idx = sorted({round(i * (len(rest) - 1) / (N_SPREAD - 1)) for i in range(N_SPREAD)})
    spread = [rest[i] for i in spread_idx]

    # preserve score-desc order, drop any accidental duplicates
    seen = set()
    ordered = []
    for cid in top + spread:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def main():
    wanted = select_corpus_ids()
    wanted_set = set(wanted)

    with open(CORPUS_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        by_id = {r["Corpus ID"]: r for r in reader if r["Corpus ID"] in wanted_set}
        fieldnames = reader.fieldnames or []

    with open(SAMPLE_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid in wanted:
            row = by_id[cid]
            row["Abstract"] = NO_ABSTRACT
            writer.writerow(row)

    print(f"Wrote {SAMPLE_FILE}: {len(wanted)} papers, abstracts removed")


if __name__ == "__main__":
    main()
