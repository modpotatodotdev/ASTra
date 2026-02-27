# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "matplotlib",
# ]
# ///

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import Counter

COLORS = ['#4f46e5', '#f59e0b', '#10b981', '#ec4899', '#8b5cf6', '#ef4444']

def parse_patch_stats(patch_text: str) -> dict:
    """Parse a unified diff to extract basic statistics."""
    lines_added = 0
    lines_removed = 0
    files_modified = set()
    
    if not patch_text or not patch_text.strip():
        return {
            "empty": True,
            "lines_added": 0,
            "lines_removed": 0,
            "files_modified": 0,
        }
        
    for line in patch_text.splitlines():
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            filename = line[6:].strip()
            files_modified.add(filename)
        elif line.startswith('+') and not line.startswith('+++'):
            lines_added += 1
        elif line.startswith('-') and not line.startswith('---'):
            lines_removed += 1
            
    return {
        "empty": False,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "files_modified": len(files_modified)
    }

def load_data(filepath: str) -> list[dict]:
    metrics = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                patch = record.get('model_patch', '')
                stats = parse_patch_stats(patch)
                stats['eval_correctness'] = record.get('eval_correctness', 'unknown')
                metrics.append(stats)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line in JSONL file: {line[:50]}...")
    return metrics

def make_dashboard(metrics: list[dict], out_path: str):
    if not metrics:
        print("No valid metrics found to visualize.")
        return

    # Extract all methods
    methods = sorted(list(set(m.get('method', 'astra') for m in metrics)))
    if 'astra' in methods:
        methods.remove('astra')
        methods = ['astra'] + methods

    # Setup the plot
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)

    # We need to compute stats grouped by method
    def get_method_metrics(method_name):
        return [m for m in metrics if m.get('method', 'astra') == method_name]

    # --- Plotting helpers ---
    def plot_grouped_bars(ax, categories, data_by_method, title, ylabel, xtick_labels=None, rotation=0, show_legend=False):
        """Helper to plot grouped bar charts for multiple methods"""
        x = np.arange(len(categories))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            counts = data_by_method[method]
            offset = x + i * width - 0.4 + width / 2
            color = COLORS[i % len(COLORS)]
            bars = ax.bar(offset, counts, width, label=method.upper(), color=color, alpha=0.9)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height + (max(max(data_by_method.values())) * 0.01),
                             f'{int(height)}', ha='center', va='bottom', fontsize=9)
                             
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels if xtick_labels else categories, rotation=rotation, ha='center' if rotation == 0 else 'right', fontsize=10)
        ax.set_title(title, fontweight='bold', pad=20, fontsize=12)
        if show_legend:
            ax.legend(title='Method', fontsize=9, loc='upper right')

    # --- Chart 1: Generation Success Rate ---
    ax0 = fig.add_subplot(gs[0, 0])
    success_data = {}
    for method in methods:
        m_data = get_method_metrics(method)
        empty = sum(1 for m in m_data if m['empty'])
        valid = len(m_data) - empty
        success_data[method] = [valid, empty]
        
    plot_grouped_bars(ax0, ['Generated Patch', 'Failed/Empty'], success_data, 
                      'Zero-Shot Patch Generation Success', 'Number of Test Cases', 
                      ['Generated Patch', 'Failed/Empty'], show_legend=True)

    # --- Chart 2: Patch Correctness Heuristic ---
    ax1 = fig.add_subplot(gs[0, 1])
    correctness_cats = ['line_overlap', 'file_overlap_only', 'no_file_overlap', 'empty']
    display_labels = [
        'High Confidence\n(Line Overlap)',
        'Partial Match\n(File Overlap)',
        'Incorrect\n(Wrong File)',
        'Empty/Failed\n(No Output)'
    ]
    
    correctness_data = {}
    for method in methods:
        m_data = get_method_metrics(method)
        counts = Counter(m['eval_correctness'] for m in m_data)
        correctness_data[method] = [counts.get(cat, 0) for cat in correctness_cats]
        
    plot_grouped_bars(ax1, correctness_cats, correctness_data, 
                      'Patch Correctness Heuristic', 'Number of Patches', display_labels, rotation=15)

    # --- Chart 3: Median Patch Size ---
    ax2 = fig.add_subplot(gs[1, 0])
    size_cats = ['Lines Added', 'Lines Removed', 'Files Modified']
    size_data = {}
    for method in methods:
        m_data = get_method_metrics(method)
        valid_m = [m for m in m_data if not m['empty']]
        if valid_m:
            med_added = np.median([m['lines_added'] for m in valid_m])
            med_removed = np.median([m['lines_removed'] for m in valid_m])
            med_files = np.median([m['files_modified'] for m in valid_m])
            size_data[method] = [med_added, med_removed, med_files]
        else:
            size_data[method] = [0, 0, 0]
            
    plot_grouped_bars(ax2, size_cats, size_data, 
                      'Median Patch Size (Valid Patches Only)', 'Count', size_cats)

    # --- Chart 4: Files Modified Distribution (1, 2, 3+) ---
    ax3 = fig.add_subplot(gs[1, 1])
    file_cats = ['1 File', '2 Files', '3+ Files']
    file_data = {}
    for method in methods:
        m_data = get_method_metrics(method)
        valid_m = [m for m in m_data if not m['empty']]
        
        c1 = sum(1 for m in valid_m if m['files_modified'] == 1)
        c2 = sum(1 for m in valid_m if m['files_modified'] == 2)
        c3 = sum(1 for m in valid_m if m['files_modified'] >= 3)
        
        file_data[method] = [c1, c2, c3]
        
    plot_grouped_bars(ax3, file_cats, file_data, 
                      'Files Modified per Patch Distribution', 'Number of Patches', file_cats)

    total_cases_per = len(metrics) // max(1, len(methods))
    fig.suptitle(f'ASTra vs Baselines: Zero-Shot Patch Generation (N={total_cases_per} / method)', 
                 fontsize=18, fontweight='900', y=0.96)

    os.makedirs(os.path.dirname(Path(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f'Wrote SVG dashboard to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help="Path to predictions.jsonl")
    parser.add_argument('--output', '-o', default='bench/predictions_report.svg', help="Output SVG path")
    args = parser.parse_args()
    
    metrics = load_data(args.input)
    make_dashboard(metrics, args.output)
