# XYbench

A benchmark for evaluating code search and retrieval systems on the **XY Problem**.

## What is the XY Problem?

The XY Problem occurs when:
- **X** = the user-reported symptom (what appears in the error/issue description)
- **Y** = the actual root cause (where the patch must land in the codebase)

The symptom and root cause are **structurally distant** — keyword overlap is low and
traversing from symptom to fix requires multiple logical hops through the call graph.
This is exactly where traditional keyword search (grep, ripgrep, BM25 RAG) fails, and
where graph-based semantic search (ASTra) succeeds.

## Dataset

Curated from [SWE-bench-lite](https://github.com/princeton-nlp/SWE-bench) (300 real-world
bug/fix pairs from open-source Python projects) using Gemini 2.5 Flash with extended
thinking.

### Qualification criteria

| Criterion | Threshold |
|-----------|-----------|
| `is_xy_problem` | `true` |
| `keyword_overlap` (1–10) | ≤ 3 |
| `logical_distance` (1–10) | ≥ 7 |

### Files

| File | Description |
|------|-------------|
| `data/xy_benchmark_50.jsonl` | 50 hardest XY cases (primary benchmark set) |
| `data/xy_benchmark_full.jsonl` | All 66 qualifying XY cases |
| `data/xy_scores_all.jsonl` | Raw LLM evaluation scores for all 2,281 SWE-bench-lite cases |

## Scripts

### `scripts/curate_xy_bench.py`

Runs the LLM-powered curation agent that analyses SWE-bench-lite cases and selects XY
problems.

```bash
uv run benchmarks/xybench/scripts/curate_xy_bench.py \
  --input benchmarks/data/swebench-lite.jsonl \
  --output data/xy_benchmark_50.jsonl \
  --limit 50
```

Requirements: `litellm`, `pydantic`, and a `GEMINI_API_KEY` (or compatible provider).

## Reports

| File | Description |
|------|-------------|
| `reports/xy_benchmark_e2e_report_new_FINAL.svg` | Visual dashboard comparing ASTra vs grep/ripgrep on 50 XY cases |
| `reports/xy_benchmark_e2e_report_final.json` | Raw JSON results for the same run |

To compare XYbench against other benchmark report JSON files using the same
flat-rate metric (`oracle_hit_rate`), use the central grid script from the repo root:

```bash
uv run --script scripts/view_benchmark_grid.py \
  --reports benchmarks/reports/swe_bench_report.json \
            benchmarks/xybench/reports/xy_benchmark_e2e_report_final.json
```

### Key results (50-case XY subset)

| Metric | ASTra | Grep | Ripgrep |
|--------|------:|-----:|--------:|
| Retrieval latency (median) | 137.4 ms | 55.5 ms | 150.9 ms |
| Oracle file hit-rate | **26 %** | 6 % | 0 % |
| Recall@k | **12.1 %** | 4.1 % | 0 % |

On hard XY problems where keywords give no signal, ASTra's graph-based semantic
traversal retrieves the correct files **26 % of the time** compared to 0 % for ripgrep.

## Migration note

This folder is being extracted into a dedicated standalone repository where ASTra will
be used as an MCP tool. The contents here represent the current snapshot prior to that
migration.
