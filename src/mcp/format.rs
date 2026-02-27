use std::collections::HashSet;

use serde::Serialize;

use crate::parser::SymbolKind;
use crate::search::{ExecutionPath, PathNode};

use super::types::FILE_SIZE_CAP;

/// Format search results into a human-readable string.
pub(crate) fn format_search_results(
    paths: &[ExecutionPath],
    return_nodes: usize,
    uncap_file_size: bool,
) -> String {
    if paths.is_empty() {
        return "No relevant execution paths found.".to_string();
    }

    let mut output = String::new();
    for (i, path) in paths.iter().enumerate() {
        output.push_str(&format!(
            "=== Execution Path {} (score: {:.3}) ===\n",
            i + 1,
            path.score
        ));

        // Show the path as a chain: fn_a -> fn_b -> fn_c
        let chain: Vec<&str> = path.nodes.iter().map(|n| n.name.as_str()).collect();
        output.push_str(&format!("Path: {}\n\n", chain.join(" → ")));

        let full_body_indices = nodes_with_full_body(&path.nodes, return_nodes);

        for (node_idx, node) in path.nodes.iter().enumerate() {
            let show_full = full_body_indices.contains(&node_idx);
            output.push_str(&format!(
                "▸ {} ({}:{}-{}) [{}]\n",
                node.name,
                node.file_path,
                node.line_range.0 + 1,
                node.line_range.1 + 1,
                if show_full { "full" } else { "skeleton" },
            ));
            let snippet = if show_full {
                cap_body(&node.body, uncap_file_size)
            } else {
                node.skeleton_context()
            };
            output.push_str(&format!("```\n{}\n```\n\n", snippet));
        }
        output.push('\n');
    }
    output
}

#[derive(Serialize)]
struct StructuredSearchResults {
    paths: Vec<StructuredExecutionPath>,
}

#[derive(Serialize)]
struct StructuredExecutionPath {
    score: f32,
    nodes: Vec<StructuredPathNode>,
}

#[derive(Serialize)]
struct StructuredPathNode {
    symbol_id: String,
    name: String,
    symbol_kind: &'static str,
    file_path: String,
    line_range: (usize, usize),
    is_terminal: bool,
    skeleton: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_body: Option<String>,
    relevance: f32,
}

pub(crate) fn format_structured_search_results(
    paths: &[ExecutionPath],
    return_nodes: usize,
    uncap_file_size: bool,
) -> String {
    let paths = paths
        .iter()
        .map(|path| {
            let full_body_indices = nodes_with_full_body(&path.nodes, return_nodes);
            StructuredExecutionPath {
                score: path.score,
                nodes: path
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(index, node)| {
                        let is_terminal = index + 1 == path.nodes.len();
                        let show_full = full_body_indices.contains(&index);
                        StructuredPathNode {
                            symbol_id: node.symbol_id.clone(),
                            name: node.name.clone(),
                            symbol_kind: symbol_kind_name(node.symbol_kind),
                            file_path: node.file_path.clone(),
                            line_range: node.line_range,
                            is_terminal,
                            skeleton: node.skeleton_context(),
                            full_body: show_full.then(|| cap_body(&node.body, uncap_file_size)),
                            relevance: node.relevance,
                        }
                    })
                    .collect(),
            }
        })
        .collect();

    serde_json::to_string_pretty(&StructuredSearchResults { paths }).unwrap()
}

fn symbol_kind_name(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Function => "function",
        SymbolKind::Method => "method",
        SymbolKind::Class => "class",
        SymbolKind::Module => "module",
        SymbolKind::Import => "import",
    }
}

/// Return the set of node indices within `nodes` that should have their full
/// body included in the response, based on the `return_nodes` parameter.
///
/// - 0: no full bodies (skeleton-only, default)
/// - 1: terminal (last) node only
/// - 2: midpoint and terminal nodes
/// - 3: start, midpoint, and terminal nodes
/// - >3: the `return_nodes` most-relevant nodes by relevance score
fn nodes_with_full_body(nodes: &[PathNode], return_nodes: usize) -> HashSet<usize> {
    if return_nodes == 0 || nodes.is_empty() {
        return HashSet::new();
    }

    let len = nodes.len();

    if return_nodes >= len {
        return (0..len).collect();
    }

    match return_nodes {
        1 => [len - 1].into(),
        2 => {
            let mid = len / 2;
            [mid, len - 1].into()
        }
        3 => {
            let mid = len / 2;
            [0, mid, len - 1].into()
        }
        n => {
            // Pick the n nodes with the highest relevance scores.
            let mut indexed: Vec<(usize, f32)> = nodes
                .iter()
                .enumerate()
                .map(|(i, node)| (i, node.relevance))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(n);
            indexed.iter().map(|(i, _)| *i).collect()
        }
    }
}

/// Truncate `body` to at most `FILE_SIZE_CAP` characters and append `...`
/// unless `uncap` is `true` or the body is already short enough.
fn cap_body(body: &str, uncap: bool) -> String {
    if uncap {
        return body.to_string();
    }
    if let Some((byte_idx, _)) = body.char_indices().nth(FILE_SIZE_CAP) {
        format!("{}...", &body[..byte_idx])
    } else {
        body.to_string()
    }
}
