"""
XY Benchmark Curation Agent
============================
Uses Gemini 2.5 Flash (with thinking) to analyze SWE-bench-lite cases and
identify the hardest XY problems — cases where the user-reported symptom (X)
is structurally distant from the actual root-cause patch (Y).

These are the cases where keyword-based RAG would FAIL, but graph-based
execution traversal (ASTra) would SUCCEED.

Usage:
    uv run benchmarks/xybench/scripts/curate_xy_bench.py [--input ../data/swebench-lite.jsonl] [--output data/xy_benchmark_50.jsonl] [--limit 50]
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from pydantic import BaseModel, Field
import litellm
from litellm import completion


# ─── Structured Output Schema ────────────────────────────────────────────────

class XYScore(BaseModel):
    """Structured evaluation of a single SWE-bench case for the XY Problem Benchmark."""
    is_xy_problem: bool = Field(
        description="True if the issue symptom (X) is structurally distant from the file patched (Y). "
                    "The user complained about one thing, but the fix was in a completely different subsystem."
    )
    keyword_overlap: int = Field(
        description="Score 1-10. How much textual overlap exists between the issue description and the "
                    "patched file/function names. 1 = zero shared keywords. 10 = the issue literally names "
                    "the exact file and function that was patched."
    )
    logical_distance: int = Field(
        description="Score 1-10. How many logical hops it takes to get from the symptom to the root cause. "
                    "1 = the patch is in the exact module the user mentioned. "
                    "10 = the patch is in a deeply nested dependency that requires multi-hop execution tracing."
    )
    reasoning: str = Field(
        description="1-2 sentences explaining why this is or isn't a good XY problem benchmark case. "
                    "Mention the gap between the symptom and the fix."
    )


# ─── Evaluation Logic ────────────────────────────────────────────────────────

def evaluate_case(case: dict, model: str) -> XYScore | None:
    """Send a single SWE-bench case to LLM for XY scoring using LiteLLM."""
    
    issue_text = case.get("problem_statement", "")[:2000]
    patch_text = case.get("patch", "")[:2000]
    instance_id = case.get("instance_id", "unknown")
    repo = case.get("repo", "unknown")

    prompt = f"""You are an expert software engineer evaluating bug reports for the "XY Problem Benchmark."

We want to find cases where a developer's REPORTED SYMPTOM (X) is structurally distant from the 
ACTUAL ROOT CAUSE (Y) — cases where standard keyword-based code search would completely fail to 
locate the relevant files, but execution-graph traversal would succeed.

Evaluate the following case:

REPOSITORY: {repo}
INSTANCE: {instance_id}

═══ ISSUE (The Symptom — X) ═══
{issue_text}

═══ PATCH (The Root Cause — Y) ═══
{patch_text}

Score this case. Be strict — we only want the brutally hard cases where there is genuine structural 
distance between what the user described and what was actually patched. If the user's issue text 
essentially names the file or function that was patched, that's a LOW logical_distance and HIGH 
keyword_overlap."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=XYScore,
                drop_params=True # Drops unsupported params like thinking_config if passed
            )
            
            content = response.choices[0].message.content
            # It's usually a JSON string mapped to the Pydantic schema
            return XYScore.model_validate_json(content)
            
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Rate limit" in err_str:
                print(f"\n  ⏳ Rate limit hit! Sleeping for 10s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(10)
                continue
            print(f"\n  ⚠ Error evaluating {instance_id}: {e}", file=sys.stderr)
            return None
    return None


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Curate XY Benchmark from SWE-bench-lite")
    parser.add_argument("--input", default="../data/swebench-lite.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="xy_benchmark_50.jsonl", help="Output JSONL file")
    parser.add_argument("--scores-output", default="xy_scores_all.jsonl", help="All scores output file")
    parser.add_argument("--limit", type=int, default=5000, help="Max XY cases to collect (set high to process entire dataset)")
    parser.add_argument("--model", type=str, default="github_copilot/gpt-5-mini", help="LiteLLM model to use (default: github_copilot/gpt-5-mini)")
    parser.add_argument(
        "--keyword-max", type=int, default=3,
        help="Max keyword_overlap score to qualify (inclusive, default: 3)"
    )
    parser.add_argument(
        "--distance-min", type=int, default=7,
        help="Min logical_distance score to qualify (inclusive, default: 7)"
    )
    args = parser.parse_args()

    # ── Validate API key ──
    # Note: litellm handles grabbing GITHUB_TOKEN or COPILOT_API_KEY from the environment
    # but we can try to load .env just in case it's stored there.
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        try:
            load_dotenv(env_path)
        except ImportError:
            pass

    # ── Load dataset ──
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent.parent / input_path

    if not input_path.exists():
        print(f"❌ Dataset not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  🤖 XY Benchmark Curation Agent                                ║")
    print(f"║  Model: {args.model:<33}                      ║")
    print(f"║  Dataset: {len(cases):<4} SWE-bench-lite cases                            ║")
    print(f"║  Filter: keyword_overlap ≤ {args.keyword_max}, logical_distance ≥ {args.distance_min}             ║")
    print(f"║  Target: {args.limit:<4} hardest XY cases                               ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    print()

    xy_benchmark = []
    all_scores = []
    
    # Load previously analyzed scores to allow resuming after rate limits
    output_dir = Path(__file__).resolve().parent.parent
    scores_path = output_dir / args.scores_output
    if scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        all_scores.append(rec)
                        if (rec["is_xy_problem"] and 
                                rec["keyword_overlap"] <= args.keyword_max and 
                                rec["logical_distance"] >= args.distance_min):
                            # Attempt to find the full case to push to xy_benchmark
                            matching_case = next((c for c in cases if c.get("instance_id") == rec["instance_id"]), None)
                            if matching_case and matching_case not in xy_benchmark:
                                xy_benchmark.append(matching_case)
                    except json.JSONDecodeError:
                        print(f"  ⚠ Corrupted JSON line found in {args.scores_output}, skipping...")
                        continue
                            
        print(f"Loaded {len(all_scores)} raw score records.")
        # Create set of normalized IDs for lookup
        analyzed_instance_ids = {s["instance_id"].strip() for s in all_scores if "instance_id" in s}
        print(f"Loaded {len(analyzed_instance_ids)} unique IDs to skip (Resuming...)")
        print(f"Sample IDs: {list(analyzed_instance_ids)[:3]}...")
        print(f"Already found {len(xy_benchmark)} qualifiers from previous rounds.\n")
    else:
        analyzed_instance_ids = set()

    # Get a list of IDs currently in the dataset for diagnostic purposes
    dataset_ids = {c.get("instance_id", "unknown").strip() for c in cases}
    overlap = analyzed_instance_ids.intersection(dataset_ids)
    print(f"Sync check: {len(overlap)} / {len(analyzed_instance_ids)} scored IDs found in THIS dataset.\n")

    start_time = time.time()
    cases_analyzed_this_run = 0

    for i, case in enumerate(cases):
        instance_id = case.get("instance_id", "unknown")
        
        if instance_id in analyzed_instance_ids:
            continue
            
        cases_analyzed_this_run += 1
        
        elapsed = time.time() - start_time
        rate = cases_analyzed_this_run / elapsed if elapsed > 0 else 0
        eta = (len(cases) - i - 1) / rate if rate > 0 else 0

        print(f"[{i+1:3d}/{len(cases)}] Analyzing {instance_id}...", end=" ", flush=True)

        score = evaluate_case(case, args.model)

        if score is None:
            print("⚠ SKIPPED")
            continue

        # Save every score for analysis
        score_record = {
            "instance_id": instance_id,
            "repo": case.get("repo", ""),
            **score.model_dump(),
        }
        all_scores.append(score_record)

        # Check if it qualifies
        qualifies = (
            score.is_xy_problem
            and score.keyword_overlap <= args.keyword_max
            and score.logical_distance >= args.distance_min
        )

        if qualifies:
            xy_benchmark.append(case)
            print(
                f"🔥 XY HIT! kw={score.keyword_overlap} dist={score.logical_distance} "
                f"| {score.reasoning[:80]}"
            )
        else:
            status = "✗" if not score.is_xy_problem else "~"
            print(
                f"{status}  kw={score.keyword_overlap} dist={score.logical_distance}"
            )

        # Progress stats
        if cases_analyzed_this_run % 25 == 0:
            print(f"\n  📊 Progress: {len(xy_benchmark)} XY cases found so far "
                  f"| {cases_analyzed_this_run} analyzed this session | Session ETA: {eta/60:.1f}m\n")

        # Save dynamically to guarantee we don't lose data if we CTRL+C
        with open(scores_path, "w", encoding="utf-8") as f:
            for record in all_scores:
                f.write(json.dumps(record) + "\n")

        if len(xy_benchmark) >= args.limit:
            print(f"\n🎯 Reached target of {args.limit} XY cases! Stopping early.")
            break

        # Re-try rate limiting is largely handled internally, but just to be nice
        time.sleep(1)

    # ── Save final curated benchmark ──
    output_path = output_dir / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        for case in xy_benchmark:
            f.write(json.dumps(case) + "\n")

    elapsed = time.time() - start_time

    print()
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  ✅ Curation Complete                                           ║")
    print(f"║  XY Cases Found: {len(xy_benchmark):3d} / {len(cases)}{' ' * (42 - len(str(len(cases))))}║")
    print(f"║  Total Time: {elapsed:.1f}s ({elapsed/60:.1f}m){' ' * max(0, 40 - len(f'{elapsed:.1f}s ({elapsed/60:.1f}m)'))}║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n  📄 Benchmark: {output_path}")
    print(f"  📊 All Scores: {scores_path}")
    print()

    # Print summary table of found cases
    if xy_benchmark:
        print("  🔥 Curated XY Benchmark Cases:")
        print("  " + "─" * 64)
        for case in xy_benchmark:
            iid = case.get("instance_id", "?")
            # Find matching score
            matching = [s for s in all_scores if s["instance_id"] == iid]
            if matching:
                s = matching[0]
                print(f"  {iid:<45} kw={s['keyword_overlap']} dist={s['logical_distance']}")
        print("  " + "─" * 64)


if __name__ == "__main__":
    main()
