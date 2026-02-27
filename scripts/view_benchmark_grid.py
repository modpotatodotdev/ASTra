# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""
Render benchmark reports in a columned grid with a unified flat-rate score.

Usage:
    uv run --script scripts/view_benchmark_grid.py \
      --reports benchmarks/reports/swe_bench_report.json \
                benchmarks/xybench/reports/xy_benchmark_e2e_report_new_final_fr_i_hope.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("astra", "grep", "ripgrep", "traditional_rag")
METHOD_LABELS = {
    "astra": "ASTra",
    "grep": "grep",
    "ripgrep": "ripgrep",
    "traditional_rag": "traditional_rag",
}


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_benchmark_name(path: Path, report: dict) -> str:
    dataset_path = str(report.get("metadata", {}).get("swe_bench_jsonl", "")).lower()
    if "xy_benchmark" in dataset_path:
        return "XYbench"
    if "swebench-django" in dataset_path:
        return "SWE-bench Django"
    if "swebench-lite" in dataset_path:
        return "SWE-bench Lite"
    return path.stem


def extract_flat_rate(report: dict, method: str) -> float:
    summary = report.get("summary", {}).get(method, {})
    value = summary.get("oracle_hit_rate", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def dedupe_suite_names(suite_entries: list[tuple[Path, str, dict]]) -> list[tuple[str, dict]]:
    counts: dict[str, int] = {}
    result: list[tuple[str, dict]] = []
    for path, name, report in suite_entries:
        counts[name] = counts.get(name, 0) + 1
        label = name if counts[name] == 1 else f"{name} ({path.stem})"
        result.append((label, report))
    return result


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    output = [fmt_row(headers), separator]
    output.extend(fmt_row(row) for row in rows)
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True, help="Benchmark report JSON paths")
    args = parser.parse_args()

    reports: list[tuple[str, dict]] = []
    suite_entries: list[tuple[Path, str, dict]] = []
    for raw_path in args.reports:
        path = Path(raw_path)
        report = load_report(path)
        suite_entries.append((path, infer_benchmark_name(path, report), report))
    reports = dedupe_suite_names(suite_entries)

    headers = ["Method"] + [name for name, _ in reports] + ["All (avg)"]
    rows: list[list[str]] = []
    for method in METHODS:
        method_scores: list[float] = []
        row = [METHOD_LABELS[method]]
        for _, report in reports:
            score = extract_flat_rate(report, method)
            method_scores.append(score)
            row.append(f"{score:.3f}")
        avg_score = sum(method_scores) / len(method_scores) if method_scores else 0.0
        row.append(f"{avg_score:.3f}")
        rows.append(row)

    print("Flat-rate score grid (oracle_hit_rate, 0..1):")
    print(format_table(headers, rows))


if __name__ == "__main__":
    main()
