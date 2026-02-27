# Zero-Shot Patch Generation: ASTra vs Baselines

## Setup

- **Script**: `src/bin/zero_shot_patch_benchmark.rs` — accepts `--method astra|grep|ripgrep|traditional_rag`
- **LLM**: `z_ai_coding/GLM-4.7` at `--concurrency 4`
- **Evaluation**: `bench/evaluate_correctness.py` — heuristic check against SWE-bench gold patches
- **Dataset**: 50 `astropy/astropy` SWE-bench cases × 4 methods = 200 total records

## Results

| Metric | ASTra | Grep | Ripgrep | Traditional RAG |
|--------|-------|------|---------|-----------------|
| Valid Patches | **43** | 39 | 40 | **43** |
| Failed/Empty | 7 | 11 | 10 | 7 |
| Line Overlap (High) | 11 | 11 | 9 | **15** |
| File Overlap Only | **20** | 17 | **19** | 18 |
| Wrong File | 12 | 11 | 12 | 10 |

**ASTra and RAG tied for highest completion rate (43/50).** Traditional RAG led on line overlap (15 vs 11), while ASTra led on partial-match file overlap (20 vs 18).

## Root Cause Analysis: Flat Helper File Problem

### Case: `astropy__astropy-14213`

RAG got `line_overlap` (perfect match). ASTra got `no_file_overlap` (wrong file entirely).

**Issue**: `np.histogram()` raises an error when `range=` is an astropy `Quantity`.

**Gold patch location**: `astropy/units/quantity_helper/function_helpers.py` — a large, flat file containing standalone function helpers like `histogram()`, `histogram2d()`, `histogramdd()`.

| | What happened |
|---|---|
| 🟢 **Traditional RAG** | Trivially retrieved `function_helpers.py` via BM25 keyword match. The file is literally full of `histogram`, `range`, `Quantity` keyword hits. LLM patched the exact right spot. |
| 🔴 **ASTra** | Semantically identified `astropy/units/quantity.py` as the core `Quantity` file (semantically correct for understanding the class), but missed `function_helpers.py` since it's a flat collection of loose functions with no dominant class structure in the AST skeleton. LLM added a `_histogram_helper` function in the wrong file. |

### Insight

ASTra's strength is navigating **deep class hierarchies** — it excels when the bug lives inside a method of a well-structured class. It is relatively weaker for **flat utility/helper files** that:
- Have no dominant class
- Match the issue description almost word-for-word on keywords
- Are conventional "dispatch table" files (like numpy function override registries)

This is precisely where lexical BM25 search wins: keyword density alone is sufficient.

## Cases Where ASTra Won Over RAG

ASTra beat RAG on 9 instances, typically involving:
- Cross-module class inheritance (e.g., `astropy-13158`, `astropy-13469`)
- Issues where the user's error message is semantically distant from the actual code location (e.g., a math error described in natural language, not in code terms)

## Potential Improvements

1. **Hybrid retrieval**: When issue text contains explicit function/method names matching the codebase, blend in a BM25 pass to catch flat helper files.
2. **Sparse-skeleton boosting**: Detect files with very few AST classes but many top-level functions, and give them a relevance boost when the issue query matches their function names.
3. **Multi-repo benchmark**: 50 cases over a single repo is noisy. A multi-repo run would reduce bias from repo-specific code organization patterns.
