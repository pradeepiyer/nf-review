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
