# /// script
# requires-python = ">=3.10"
# dependencies =[
#   "numpy",
#   "matplotlib",
# ]
# ///

import argparse
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

DISPLAY_NAMES =['ASTra\n(Graph+Vector)', 'Grep', 'Ripgrep', 'Traditional\nRAG']
METHOD_KEYS =['astra', 'grep', 'ripgrep', 'traditional_rag']
COLORS =['#4f46e5', '#f59e0b', '#10b981', '#ec4899'] # Adjusted for slightly deeper academic colors

def load_report(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_metrics(data: dict):
    summary = data.get('summary', {})
    
    metrics = {
        'hit_rate': [],
        'recall':[],
        'retrieval_avg':[],
        'e2e_avg':[],
        'avg_total_tokens':[],
        'avg_skeleton_tokens':[],
        'avg_body_tokens':[]
    }

    for k in METHOD_KEYS:
        method_data = summary.get(k, {})
        runs = method_data.get('runs', 1)
        if runs == 0: runs = 1

        avg_total = method_data.get('prompt_tokens', 0) / runs
        avg_skel = method_data.get('total_skeleton_tokens', 0) / runs
        # We derive body tokens by subtracting skeleton from total prompt tokens
        avg_body = avg_total - avg_skel

        metrics['hit_rate'].append(method_data.get('oracle_hit_rate', 0) * 100)
        metrics['recall'].append(method_data.get('recall_at_k', 0) * 100)
        metrics['retrieval_avg'].append(method_data.get('retrieval_avg_ms', 0))
        metrics['e2e_avg'].append(method_data.get('total_avg_ms', 0) / 1000.0) 
        metrics['avg_total_tokens'].append(avg_total)
        metrics['avg_skeleton_tokens'].append(avg_skel)
        metrics['avg_body_tokens'].append(avg_body)

    return metrics

def make_academic_svg(data: dict, out_path: str):
    metrics = extract_metrics(data)
    x = np.arange(len(DISPLAY_NAMES))

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    def add_labels(ax, bars, fmt='{:.1f}', y_offset=0):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height + y_offset,
                        fmt.format(height), ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', color='#333333')

    # --- Chart 1: Oracle Hit Rate ---
    ax0 = fig.add_subplot(gs[0, 0])
    bars0 = ax0.bar(x, metrics['hit_rate'], color=COLORS)
    ax0.set_title('Oracle Hit Rate (Multi-Hop Bugs)', fontweight='bold')
    ax0.set_ylabel('Success Rate (%)')
    ax0.set_ylim(0, max(metrics['hit_rate']) * 1.3)
    ax0.set_xticks(x); ax0.set_xticklabels(DISPLAY_NAMES)
    add_labels(ax0, bars0, '{:.1f}%', y_offset=0.5)

    # --- Chart 2: Recall@K ---
    ax1 = fig.add_subplot(gs[0, 1])
    bars1 = ax1.bar(x, metrics['recall'], color=COLORS, alpha=0.9)
    ax1.set_title('Recall@K (Path File Discovery)', fontweight='bold')
    ax1.set_ylabel('Recall (%)')
    ax1.set_ylim(0, max(metrics['recall']) * 1.3)
    ax1.set_xticks(x); ax1.set_xticklabels(DISPLAY_NAMES)
    add_labels(ax1, bars1, '{:.1f}%', y_offset=0.5)

    # --- Chart 3: Context Payload Breakdown (THE NEW CHART) ---
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Plot Bottom Stack: Structural Mapping (Skeletons)
    p_skel = ax2.bar(x, metrics['avg_skeleton_tokens'], color='#111827', label='Discovery (Structural Map)')
    # Plot Top Stack: Raw Code Bodies
    p_body = ax2.bar(x, metrics['avg_body_tokens'], bottom=metrics['avg_skeleton_tokens'], color=COLORS, alpha=0.75, label='Extraction (Raw Code)')
    
    ax2.set_title('Payload Breakdown: Discovery vs. Extraction', fontweight='bold')
    ax2.set_ylabel('Avg Tokens per Query')
    ax2.set_ylim(0, max(metrics['avg_total_tokens']) * 1.3)
    ax2.set_xticks(x); ax2.set_xticklabels(DISPLAY_NAMES)
    
    # Add a special label just to highlight the tiny 15 tokens for ASTra
    astra_skel = metrics['avg_skeleton_tokens'][0]
    ax2.text(0, astra_skel + 200, f'{astra_skel:.1f} tokens', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#111827', style='italic')
    
    # Label the total top heights
    add_labels(ax2, p_body, '{:.0f}')
    
    # Add a custom legend to explain the stacks
    from matplotlib.patches import Patch
    legend_elements =[
        Patch(facecolor='#111827', label='Discovery Map (Signatures)'),
        Patch(facecolor='#9ca3af', alpha=0.75, label='Extraction Payload (Target Body)')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # --- Chart 4: Retrieval Latency ---
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(x, metrics['retrieval_avg'], color=COLORS)
    ax3.set_title('Engine Retrieval Latency', fontweight='bold')
    ax3.set_ylabel('Milliseconds (ms)')
    ax3.set_ylim(0, max(metrics['retrieval_avg']) * 1.3)
    ax3.set_xticks(x); ax3.set_xticklabels(DISPLAY_NAMES)
    add_labels(ax3, bars3, '{:.1f} ms')

    # --- Chart 5: End-to-End Pipeline Latency ---
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(x, metrics['e2e_avg'], color=COLORS, alpha=0.9)
    ax4.set_title('Total E2E Pipeline Time (w/ LLM)', fontweight='bold')
    ax4.set_ylabel('Seconds (s)')
    ax4.set_ylim(0, max(metrics['e2e_avg']) * 1.3)
    ax4.set_xticks(x); ax4.set_xticklabels(DISPLAY_NAMES)
    add_labels(ax4, bars4, '{:.1f} s')

    # --- Chart 6: Context Cost vs Accuracy Scatter ---
    ax5 = fig.add_subplot(gs[1, 2])
    for i in range(len(METHOD_KEYS)):
        ax5.scatter(metrics['avg_total_tokens'][i], metrics['hit_rate'][i], 
                    color=COLORS[i], s=200, label=DISPLAY_NAMES[i].replace('\n', ' '), edgecolors='black', zorder=5)
    
    # Add special marker for ASTra Signature (Discovery) cost vs same hit rate
    ax5.scatter(metrics['avg_skeleton_tokens'][0], metrics['hit_rate'][0], 
                color='#111827', marker='*', s=350, label='ASTra (Discovery Map Only)', edgecolors='white', linewidth=1, zorder=6)
    
    # Add a connecting line between ASTra Total and ASTra Discovery to show the "Payload Gap"
    ax5.plot([metrics['avg_skeleton_tokens'][0], metrics['avg_total_tokens'][0]], 
             [metrics['hit_rate'][0], metrics['hit_rate'][0]], 
             color='#4f46e5', linestyle='--', linewidth=1, alpha=0.5, zorder=4)

    ax5.set_title('Total Context Cost vs Accuracy', fontweight='bold')
    ax5.set_xlabel('Avg Tokens per Query')
    ax5.set_ylabel('Oracle Hit Rate (%)')
    ax5.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax5.legend(loc='lower right', fontsize=7)

    fig.suptitle('XYBench (Multi-Hop) Evaluation Results', fontsize=18, fontweight='900', y=0.98)
    
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f'Wrote academic SVG report to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default='benchmarks/reports/academic_benchmark.svg')
    args = parser.parse_args()
    data = load_report(args.input)
    make_academic_svg(data, args.output)
