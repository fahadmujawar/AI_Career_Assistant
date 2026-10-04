r"""Retrieval test: does retrieve() find the expected reference file?

No API calls. Needs the search index (python -m rag.ingest, python -m rag.index).

Run from anywhere:
    python eval\eval_retrieval.py
    python eval\eval_retrieval.py --cases other_cases.json --no-write
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # rag/ uses paths relative to the project root
sys.path.insert(0, str(ROOT))

from results_md import clean, write_section  # noqa: E402
from rag.index import INDEX_PATH, META_PATH, MODEL_NAME  # noqa: E402
from rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, KB_DIR  # noqa: E402

K = 4                   # the app searches the reference documents with k=4
DOC_TYPE = "reference"


def load_cases(path):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(cases, list) or not cases:
        sys.exit(f"{path} must be a non-empty JSON list of cases.")

    for i, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or not {"query", "expected_file"} <= set(case):
            sys.exit(f"Case {i} in {path} needs both \"query\" and \"expected_file\".")
        if "REPLACE" in case["query"] or "REPLACE" in case["expected_file"]:
            sys.exit(
                f"Case {i} in {path} is still the placeholder (it says \"REPLACE\").\n"
                "Write your own cases in that file, then run this script again."
            )
        if not (KB_DIR / case["expected_file"]).is_file():
            sys.exit(
                f"Case {i}: expected_file \"{case['expected_file']}\" does not exist "
                f"under {KB_DIR}/ (use a path like reference/ats_keyword_guide.txt)."
            )

    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default="eval/retrieval_cases.json")
    parser.add_argument("--no-write", action="store_true",
                        help="print results only; do not update eval/results.md")
    args = parser.parse_args()

    cases = load_cases(args.cases)

    if not INDEX_PATH.exists() or not META_PATH.exists():
        sys.exit("Search index not found. Run python -m rag.ingest and "
                 "python -m rag.index from the project root first.")

    from rag.retrieve import retrieve  # loads the embedding model

    rows = []
    for case in cases:
        results = retrieve(case["query"], k=K, doc_type=DOC_TYPE)
        sources = [r["source"] for r in results]
        rank = sources.index(case["expected_file"]) + 1 if case["expected_file"] in sources else None
        rows.append({
            "query": case["query"],
            "expected": case["expected_file"],
            "rank": rank,
            "hit1": rank == 1,
            "hit4": rank is not None and rank <= K,
            "top": sources[0] if sources else "-",
        })

    n = len(rows)
    hit1 = sum(r["hit1"] for r in rows)
    hit4 = sum(r["hit4"] for r in rows)

    lines = [
        "## Retrieval",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Embedding model: {MODEL_NAME}",
        f"- Chunk size: {CHUNK_SIZE} words, overlap: {CHUNK_OVERLAP} words",
        f"- k: {K}, documents searched: {DOC_TYPE}",
        f"- Number of cases: {n}",
        "",
        "| # | Query | Expected file | Rank | hit@1 | hit@4 | Top result |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {clean(r['query'], 120)} | {r['expected']} | "
            f"{r['rank'] or 'not found'} | {'yes' if r['hit1'] else 'no'} | "
            f"{'yes' if r['hit4'] else 'no'} | {r['top']} |"
        )
    lines += [
        "",
        f"**hit@1: {hit1}/{n}   hit@4: {hit4}/{n}**",
        "",
        "- hit@1: the expected file was the very first result.",
        "- hit@4: the expected file was somewhere in the top 4 results, "
        "which is what the app actually sends to the AI.",
        "- Rank counts chunks, not files: two chunks of another file ahead "
        "of the expected one put it at rank 3.",
    ]
    report = "\n".join(lines)

    print(report)
    if args.no_write:
        print("\n(--no-write: eval/results.md not updated)")
    else:
        write_section("retrieval", report)
        print("\nWrote eval/results.md")


if __name__ == "__main__":
    main()
