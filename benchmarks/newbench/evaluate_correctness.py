import json
import argparse
import re
from pathlib import Path

def parse_diff(diff_str):
    """Parses a unified diff and returns a dict mapping filenames to sets of added and removed lines."""
    files = {}
    current_file = None
    
    for line in diff_str.splitlines():
        if line.startswith("--- a/") or line.startswith("--- a"):
            continue
        elif line.startswith("+++ b/") or line.startswith("+++ b"):
            current_file = line[6:].strip()
            files[current_file] = {"added": set(), "removed": set()}
        elif current_file:
            if line.startswith("+") and not line.startswith("+++"):
                files[current_file]["added"].add(line[1:].strip())
            elif line.startswith("-") and not line.startswith("---"):
                files[current_file]["removed"].add(line[1:].strip())
                
    return files

def evaluate(gold_diff, model_diff):
    if not model_diff:
        return "empty"
        
    gold_files = parse_diff(gold_diff)
    model_files = parse_diff(model_diff)
    
    gold_file_set = set(gold_files.keys())
    model_file_set = set(model_files.keys())
    
    if not gold_file_set.intersection(model_file_set):
        return "no_file_overlap"
        
    # Check for line overlap
    # We say there is line overlap if any added/removed lines in the model patch match the gold patch
    for file in gold_file_set.intersection(model_file_set):
        g_added = gold_files[file]["added"]
        m_added = model_files[file]["added"]
        
        g_removed = gold_files[file]["removed"]
        m_removed = model_files[file]["removed"]
        
        # Remove empty lines for comparison
        g_added = {l for l in g_added if l}
        m_added = {l for l in m_added if l}
        g_removed = {l for l in g_removed if l}
        m_removed = {l for l in m_removed if l}
        
        if g_added.intersection(m_added) or g_removed.intersection(m_removed):
            return "line_overlap"
            
    return "file_overlap_only"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.jsonl")
    parser.add_argument("--swe-bench", type=str, required=True, help="Path to swe-bench-lite.jsonl or full")
    parser.add_argument("--output", type=str, required=True, help="Path to output evaluated predictions.jsonl")
    args = parser.parse_args()

    # Load SWE-bench data
    swe_bench = {}
    with open(args.swe_bench, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            swe_bench[data["instance_id"]] = data

    evaluated = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            pred = json.loads(line)
            instance_id = pred["instance_id"]
            if instance_id not in swe_bench:
                print(f"Warning: {instance_id} not found in SWE-bench data")
                continue
                
            gold_patch = swe_bench[instance_id]["patch"]
            model_patch = pred.get("model_patch", "")
            
            correctness = evaluate(gold_patch, model_patch)
            pred["eval_correctness"] = correctness
            evaluated.append(pred)

    with open(args.output, "w", encoding="utf-8") as f:
        for pred in evaluated:
            f.write(json.dumps(pred) + "\n")
            
    print(f"Evaluated {len(evaluated)} predictions and wrote to {args.output}")

if __name__ == "__main__":
    main()
