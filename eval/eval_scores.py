r"""Score consistency (7b) and tailoring number check (7c).

MAKES REAL API CALLS: about 10 analysis calls + 2 tailoring calls
(more if a model falls back to the other one). Stops early if calls are
rate limited twice in a row.

Run from anywhere:
    python eval\eval_scores.py --dry-run      # show the plan, no API calls
    python eval\eval_scores.py
    python eval\eval_scores.py --cv knowledge_base\cvs\cv_ml_general.txt
"""

import argparse
import os
import re
import sys
import time
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # rag/ uses paths relative to the project root
sys.path.insert(0, str(ROOT))

from results_md import clean, write_section  # noqa: E402
from utils import ai_analysis  # noqa: E402
from utils.ai_analysis import analyze_cv_against_jd, tailor_cv_to_jd  # noqa: E402
from utils.ai_client import GEMINI_MODEL, GROQ_MODEL  # noqa: E402
from utils.resume_parser import parse_resume  # noqa: E402

MODELS = {"Gemini": GEMINI_MODEL, "Groq": GROQ_MODEL}
RUNS_PER_MODEL = 5
WAIT_SECONDS = 5
RATE_LIMIT = re.compile(r"429|RESOURCE_EXHAUSTED|rate.?limit", re.IGNORECASE)

# Matches 72,297  72297  0.438  .438  65%  2023
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?|\.\d+%?")


class RateLimitStop(Exception):
    pass


# Record every provider attempt (including fallbacks) without changing the
# app's code, so we can see when the requested model was rate limited.
provider_log = []
_original_call_provider = ai_analysis._call_provider


def _recording_call_provider(provider, prompt, json_mode=False):
    text, error = _original_call_provider(provider, prompt, json_mode=json_mode)
    provider_log.append({"provider": provider, "error": error})
    return text, error


ai_analysis._call_provider = _recording_call_provider


class Runner:
    """Waits between calls, times them and stops on repeated rate limits."""

    def __init__(self, sections, jd):
        self.sections, self.jd = sections, jd
        self.calls = 0
        self.rate_limit_streak = 0

    def call(self, fn, provider):
        if self.calls:
            time.sleep(WAIT_SECONDS)
        self.calls += 1

        start = len(provider_log)
        t0 = time.perf_counter()
        result = fn(self.sections, self.jd, provider=provider)
        seconds = time.perf_counter() - t0

        requested_error = next(
            (a["error"] for a in provider_log[start:]
             if a["provider"] == provider and a["error"]),
            None,
        )
        rate_limited = bool(requested_error and RATE_LIMIT.search(requested_error))
        self.rate_limit_streak = self.rate_limit_streak + 1 if rate_limited else 0

        meta = {
            "provider_used": result.pop("_provider_used", None),
            "fell_back": result.pop("_fell_back", None),
            "seconds": seconds,
            "error": result.get("error") or requested_error,
            "rate_limited": rate_limited,
        }
        result.pop("_provider_requested", None)
        return result, meta

    def check_rate_limit(self):
        if self.rate_limit_streak >= 2:
            raise RateLimitStop("Rate limited twice in a row.")


def numbers_in(text):
    """Map each number's value to how it was written. 72,297 == 72297, 65% == 65."""
    found = {}
    for token in NUMBER.findall(text):
        token = token.rstrip(",")
        try:
            value = Decimal(token.replace(",", "").rstrip("%"))
        except InvalidOperation:
            continue
        found.setdefault(value, token)
    return found


def new_numbers(original, rewritten):
    before = numbers_in(original)
    return [token for value, token in numbers_in(rewritten).items() if value not in before]


def run_scores(runner, score_rows):
    for provider in MODELS:
        for run in range(1, RUNS_PER_MODEL + 1):
            result, meta = runner.call(analyze_cv_against_jd, provider)
            fallback = meta["provider_used"] not in (None, provider)
            score_rows.append({
                "model": provider,
                "run": run,
                "score": None if "error" in result else result.get("match_score"),
                "fallback": fallback,
                "failed": "error" in result,
                **meta,
            })
            print(f"  {provider} run {run}: score={score_rows[-1]['score']} "
                  f"used={meta['provider_used']} fell_back={meta['fell_back']} "
                  f"{meta['seconds']:.1f}s"
                  + (f"  error: {clean(meta['error'], 100)}" if meta["error"] else ""))
            runner.check_rate_limit()


def run_tailoring(runner, tailoring_rows):
    for provider in MODELS:
        result, meta = runner.call(tailor_cv_to_jd, provider)
        flagged, bullets = [], 0
        if "error" not in result:
            for section, items in result.items():
                for item in items:
                    bullets += 1
                    added = new_numbers(item["original"], item["rewritten"])
                    if added:
                        flagged.append((section, added, item["rewritten"]))
        tailoring_rows.append({
            "model": provider,
            "fallback": meta["provider_used"] not in (None, provider),
            "failed": "error" in result,
            "bullets": bullets,
            "flagged": flagged,
            **meta,
        })
        print(f"  {provider} tailoring: used={meta['provider_used']} "
              f"bullets={bullets} with new numbers={len(flagged)} {meta['seconds']:.1f}s"
              + (f"  error: {clean(meta['error'], 100)}" if meta["error"] else ""))
        runner.check_rate_limit()


def build_report(args, score_rows, tailoring_rows, stopped):
    api_calls = len(provider_log)
    lines = [
        "## Score consistency and tailoring number check",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Models: Gemini = `{GEMINI_MODEL}`, Groq = `{GROQ_MODEL}` (from utils/ai_client.py)",
        f"- CV: `{Path(args.cv).as_posix()}`",
        f"- JD: `{Path(args.jd).as_posix()}`",
        f"- {RUNS_PER_MODEL} analysis runs per model, {WAIT_SECONDS} s wait between calls; "
        f"API requests made: {api_calls} (fallback attempts included)",
    ]
    if stopped:
        lines += ["", f"**STOPPED EARLY: {stopped} Results below are incomplete.**"]

    lines += [
        "",
        "### Score runs (7b)",
        "",
        "| Model asked | Run | Score | provider_used | fell_back | Time (s) | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in score_rows:
        note = "fallback, excluded" if r["fallback"] else ""
        if r["failed"]:
            note = "failed: " + clean(r["error"], 150)
        elif r["rate_limited"]:
            note = (note + "; " if note else "") + "requested model rate limited"
        lines.append(
            f"| {r['model']} | {r['run']} | {r['score'] if r['score'] is not None else '-'} | "
            f"{r['provider_used'] or '-'} | {r['fell_back']} | {r['seconds']:.1f} | {note} |"
        )

    lines += [
        "",
        "| Model | Valid runs | Min | Max | Average |",
        "|---|---|---|---|---|",
    ]
    for provider in MODELS:
        scores = [r["score"] for r in score_rows
                  if r["model"] == provider and not r["fallback"] and not r["failed"]]
        if scores:
            lines.append(f"| {provider} | {len(scores)} | {min(scores)} | {max(scores)} | "
                         f"{sum(scores) / len(scores):.1f} |")
        else:
            lines.append(f"| {provider} | 0 | - | - | - |")

    lines += [
        "",
        "- Runs where the asked model was unavailable and the other model answered "
        "are marked as fallback and left out of that model's numbers.",
        f"- {RUNS_PER_MODEL} runs per model show how much the score spreads from run to "
        "run; they are not a statistical test.",
        "",
        "### Tailoring: numbers in the rewrite that are not in the original (7c)",
        "",
        "Numbers are compared by value: 72,297 = 72297, 0.438 = .438, 65% = 65.",
        "",
    ]
    for r in tailoring_rows:
        header = (f"**{r['model']}** (used {r['provider_used'] or '-'}, fell_back "
                  f"{r['fell_back']}, {r['seconds']:.1f} s)")
        if r["failed"]:
            lines += [f"- {header}: failed: {clean(r['error'], 150)}"]
        elif not r["flagged"]:
            lines += [f"- {header}: {r['bullets']} rewrites checked, no new numbers."]
        else:
            lines += [f"- {header}: {r['bullets']} rewrites checked, "
                      f"{len(r['flagged'])} with new numbers:"]
            for section, added, rewritten in r["flagged"]:
                lines.append(f"  - {section}: {', '.join(added)} in \"{clean(rewritten, 160)}\"")
    if not tailoring_rows:
        lines.append("- Not run.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cv", default="knowledge_base/cvs/cv_mle.txt")
    parser.add_argument("--jd", default="knowledge_base/jds/ai_ml_eng.txt")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without any API calls")
    args = parser.parse_args()

    for path in (args.cv, args.jd):
        if not Path(path).is_file():
            sys.exit(f"File not found: {path}")

    analysis_calls = RUNS_PER_MODEL * len(MODELS)
    print(f"CV: {args.cv}\nJD: {args.jd}")
    print(f"Models: Gemini = {GEMINI_MODEL}, Groq = {GROQ_MODEL}")
    print(f"Plan: {analysis_calls} analysis calls + {len(MODELS)} tailoring calls, "
          f"{WAIT_SECONDS} s apart (fallbacks add calls).")
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        print(f"{name} set in environment/.env: {'yes' if os.environ.get(name) else 'no'}")
    if args.dry_run:
        print("Dry run: no API calls made, eval/results.md not changed.")
        return

    sections = parse_resume(Path(args.cv).read_text(encoding="utf-8"))["sections"]
    jd = Path(args.jd).read_text(encoding="utf-8")
    runner = Runner(sections, jd)

    score_rows, tailoring_rows, stopped = [], [], None
    try:
        print("\nScore runs (7b):")
        run_scores(runner, score_rows)
        print("\nTailoring (7c):")
        run_tailoring(runner, tailoring_rows)
    except RateLimitStop as e:
        stopped = str(e)

    report = build_report(args, score_rows, tailoring_rows, stopped)
    write_section("scores", report)
    print("\nWrote eval/results.md")

    if stopped:
        print(f"\nSTOPPED: {stopped} Check your quotas before running again.")
        sys.exit(2)


if __name__ == "__main__":
    main()
