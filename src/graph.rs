use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use crate::parser::{ParsedFile, Symbol, SymbolKind};
use crate::skeleton::build_skeleton_context;

/// A node in the call graph, representing a code symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique symbol ID (`file_path::name`).
    pub symbol_id: String,
    /// Human-readable name.
    pub name: String,
    /// Kind of symbol.
    pub kind: SymbolKind,
    /// File path relative to workspace.
    pub file_path: String,
    /// Line range.
    pub line_range: (usize, usize),
    /// Source code body.
    pub body: String,
    #[serde(default)]
    pub skeleton: String,
}

/// Edge weight in the call graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Function A calls function B.
    Calls,
    /// Symbol A is contained in scope B.
    ContainedIn,
}

/// The in-memory call graph built from parsed source files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraph {
    /// The directed graph.
    pub graph: DiGraph<GraphNode, EdgeKind>,
    /// Map from symbol ID to node index for fast lookup.
    #[serde(skip)]
    pub id_to_index: HashMap<String, NodeIndex>,
    /// Map from symbol name to list of node indices (for call resolution).
    #[serde(skip)]
    pub name_to_indices: HashMap<String, Vec<NodeIndex>>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_to_index: HashMap::new(),
            name_to_indices: HashMap::new(),
        }
    }

    /// Build or extend the call graph from a set of parsed files.
    pub fn build_from_files(&mut self, parsed_files: &[ParsedFile]) {
        // Phase 1: Add all symbols as nodes
        for file in parsed_files {
            for symbol in &file.symbols {
                self.add_symbol(symbol);
            }
        }

        // Phase 2: Add edges for calls
        for file in parsed_files {
            for symbol in &file.symbols {
                if let Some(&caller_idx) = self.id_to_index.get(&symbol.id) {
                    // Add call edges
                    for call_name in &symbol.calls {
                        if let Some(callee_indices) = self.name_to_indices.get(call_name) {
                            for &callee_idx in callee_indices {
                                if caller_idx != callee_idx {
                                    let same_file = self.graph[caller_idx].file_path
                                        == self.graph[callee_idx].file_path;
                                    // Allow cross-file edges when the callee name
                                    // resolves unambiguously (only one symbol with that
                                    // name exists). When multiple symbols share a name,
                                    // restrict to same-file to avoid false positives.
                                    if same_file || callee_indices.len() == 1 {
                                        self.graph.add_edge(
                                            caller_idx,
                                            callee_idx,
                                            EdgeKind::Calls,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Add containment edges (method -> class)
                    if let Some(ref parent) = symbol.parent_scope {
                        if let Some(parent_indices) = self.name_to_indices.get(parent) {
                            for &parent_idx in parent_indices {
                                self.graph
                                    .add_edge(caller_idx, parent_idx, EdgeKind::ContainedIn);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Add a single symbol as a node.
    fn add_symbol(&mut self, symbol: &Symbol) {
        // Skip if already added
        if self.id_to_index.contains_key(&symbol.id) {
            return;
        }

        let node = GraphNode {
            symbol_id: symbol.id.clone(),
            name: symbol.name.clone(),
            kind: symbol.kind,
            file_path: symbol.file_path.clone(),
            line_range: symbol.line_range,
            body: symbol.body.clone(),
            skeleton: build_skeleton_context(&symbol.body),
        };
        let idx = self.graph.add_node(node);
        self.id_to_index.insert(symbol.id.clone(), idx);
        self.name_to_indices
            .entry(symbol.name.clone())
            .or_default()
            .push(idx);
    }

    /// Remove all symbols from a given file (for incremental updates).
    pub fn remove_file(&mut self, file_path: &str) {
        let to_remove: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|&idx| self.graph[idx].file_path == file_path)
            .collect();

        for idx in to_remove.iter().rev() {
            if let Some(node) = self.graph.node_weight(*idx) {
                self.id_to_index.remove(&node.symbol_id);
                if let Some(indices) = self.name_to_indices.get_mut(&node.name) {
                    indices.retain(|&i| i != *idx);
                }
            }
            self.graph.remove_node(*idx);
        }

        // Rebuild index maps since node indices may have shifted
        self.rebuild_indices();
    }

    /// Rebuild the id_to_index and name_to_indices maps after node removals.
    fn rebuild_indices(&mut self) {
        self.id_to_index.clear();
        self.name_to_indices.clear();
        for idx in self.graph.node_indices() {
            let node = &self.graph[idx];
            self.id_to_index.insert(node.symbol_id.clone(), idx);
            self.name_to_indices
                .entry(node.name.clone())
                .or_default()
                .push(idx);
        }
    }

    /// Get node by symbol ID.
    pub fn get_node(&self, symbol_id: &str) -> Option<(NodeIndex, &GraphNode)> {
        self.id_to_index
            .get(symbol_id)
            .map(|&idx| (idx, &self.graph[idx]))
    }

    /// Get all callers of a node (incoming call edges).
    pub fn callers(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .neighbors_directed(idx, Direction::Incoming)
            .collect()
    }

    /// Get all callees of a node (outgoing call edges).
    pub fn callees(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .neighbors_directed(idx, Direction::Outgoing)
            .collect()
    }

    /// Get all node indices.
    pub fn all_node_indices(&self) -> Vec<NodeIndex> {
        self.graph.node_indices().collect()
    }

    /// Rebuild lookup indices after deserialization.
    pub fn rebuild_after_deserialize(&mut self) {
        let indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        for idx in indices {
            if self.graph[idx].skeleton.is_empty() {
                let body = self.graph[idx].body.clone();
                self.graph[idx].skeleton = build_skeleton_context(&body);
            }
        }
        self.rebuild_indices();
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl Default for CallGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::CallGraph;
    use crate::parser::{parse_source, Language};

    #[test]
    fn test_build_graph_from_rust_source() {
        let source = r#"
fn router() {
    auth_middleware();
}

fn auth_middleware() {
    db_insert();
}

fn db_insert() {
    println!("inserting");
}
"#;
        let parsed = parse_source(source, Language::Rust, "main.rs").unwrap();
        let mut graph = CallGraph::new();
        graph.build_from_files(&[parsed]);

        assert_eq!(graph.node_count(), 3);
        // router calls auth_middleware
        let router_idx = graph.id_to_index["main.rs::router"];
        let callees = graph.callees(router_idx);
        assert_eq!(callees.len(), 1);
        assert_eq!(graph.graph[callees[0]].name, "auth_middleware");
    }

    #[test]
    fn test_remove_file() {
        let source = r#"
fn hello() {}
fn world() { hello(); }
"#;
        let parsed = parse_source(source, Language::Rust, "test.rs").unwrap();
        let mut graph = CallGraph::new();
        graph.build_from_files(&[parsed]);
        assert_eq!(graph.node_count(), 2);

        graph.remove_file("test.rs");
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_skeleton_is_precomputed() {
        let source = r#"
fn compute_total(items: Vec<i32>) -> i32 {
    // Adds all values
    items.iter().sum()
}
"#;
        let parsed = parse_source(source, Language::Rust, "calc.rs").unwrap();
        let mut graph = CallGraph::new();
        graph.build_from_files(&[parsed]);

        let idx = graph.id_to_index["calc.rs::compute_total"];
        let skeleton = &graph.graph[idx].skeleton;
        assert!(skeleton.contains("fn compute_total(items: Vec<i32>) -> i32 {"));
        assert!(skeleton.contains("// Adds all values"));
        assert!(!skeleton.contains("items.iter().sum()"));
    }
}
