# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

import json
import os

input_path = "benchmarks/data/swebench-lite.jsonl"
output_path = "benchmarks/data/swebench-django.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

django_cases = []
repo_counts = {}

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        repo = row.get("repo", "")
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if repo == "django/django":
            django_cases.append(row)

print("Repo distribution in SWE-bench-lite:")
for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
    print(f"  {repo}: {count} cases")

with open(output_path, "w", encoding="utf-8") as f:
    for row in django_cases:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nWrote {len(django_cases)} Django cases to {output_path}")
