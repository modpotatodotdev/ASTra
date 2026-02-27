# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""
Compare two SWE-bench benchmark JSON reports.

Usage:
    uv run --script scripts/compare_benchmark_reports.py \
      --before benchmarks/reports/swe_bench_django_report.full.json \
      --after benchmarks/reports/swe_bench_django_report.after_fix.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


METHODS = ["astra", "grep", "ripgrep", "traditional_rag"]
METRICS = [
    "retrieval_avg_ms",
    "retrieval_p95_ms",
    "total_avg_ms",
    "total_p95_ms",
    "total_tokens",
    "oracle_hit_rate",
]


@dataclass
class Delta:
    before: float
    after: float

    @property
    def abs_change(self) -> float:
        return self.after - self.before

    @property
    def pct_change(self) -> float:
        if self.before == 0:
            return 0.0
        return (self.abs_change / self.before) * 100.0


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: float, metric: str) -> str:
    if metric.endswith("_ms"):
        return f"{value:.3f}"
    if metric == "oracle_hit_rate":
        return f"{value * 100.0:.2f}%"
    return f"{value:.0f}"


def metric_delta(before: dict, after: dict, method: str, metric: str) -> Delta:
    b = float(before.get("summary", {}).get(method, {}).get(metric, 0.0))
    a = float(after.get("summary", {}).get(method, {}).get(metric, 0.0))
    return Delta(before=b, after=a)


def print_method_block(before: dict, after: dict, method: str) -> None:
    print(f"\n=== {method} ===")
    for metric in METRICS:
        delta = metric_delta(before, after, method, metric)
        direction = "↓" if delta.abs_change < 0 else "↑"
        if metric == "oracle_hit_rate":
            direction = "↑" if delta.abs_change > 0 else "↓"
        print(
            f"- {metric:18s}  {fmt(delta.before, metric):>10} -> {fmt(delta.after, metric):>10}  "
            f"({direction} {delta.abs_change:+.4f}, {delta.pct_change:+.2f}%)"
        )


def print_highlights(before: dict, after: dict) -> None:
    astra_tokens = metric_delta(before, after, "astra", "total_tokens")
    astra_hit = metric_delta(before, after, "astra", "oracle_hit_rate")
    astra_latency = metric_delta(before, after, "astra", "retrieval_avg_ms")

    print("\n=== highlights ===")
    print(
        f"astra total_tokens: {fmt(astra_tokens.before, 'total_tokens')} -> {fmt(astra_tokens.after, 'total_tokens')} "
        f"({astra_tokens.abs_change:+.0f}, {astra_tokens.pct_change:+.2f}%)"
    )
    print(
        f"astra oracle_hit_rate: {fmt(astra_hit.before, 'oracle_hit_rate')} -> {fmt(astra_hit.after, 'oracle_hit_rate')} "
        f"({astra_hit.abs_change:+.4f}, {astra_hit.pct_change:+.2f}%)"
    )
    print(
        f"astra retrieval_avg_ms: {fmt(astra_latency.before, 'retrieval_avg_ms')} -> {fmt(astra_latency.after, 'retrieval_avg_ms')} "
        f"({astra_latency.abs_change:+.3f}, {astra_latency.pct_change:+.2f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="Path to baseline benchmark JSON")
    parser.add_argument("--after", required=True, help="Path to new benchmark JSON")
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)

    before = load_json(before_path)
    after = load_json(after_path)

    print(f"before: {before_path}")
    print(f"after:  {after_path}")

    for method in METHODS:
        print_method_block(before, after, method)

    print_highlights(before, after)


if __name__ == "__main__":
    main()
