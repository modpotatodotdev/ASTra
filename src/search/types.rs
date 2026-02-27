use serde::{Deserialize, Serialize};

use crate::parser::SymbolKind;
use crate::skeleton::build_skeleton_context;

/// A path through the execution graph returned by search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPath {
    /// Ordered list of nodes in the path.
    pub nodes: Vec<PathNode>,
    /// Overall relevance score.
    pub score: f32,
}

/// A single node in the execution path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathNode {
    pub symbol_id: String,
    pub name: String,
    pub symbol_kind: SymbolKind,
    pub file_path: String,
    pub line_range: (usize, usize),
    pub body: String,
    #[serde(default)]
    pub skeleton: String,
    pub relevance: f32,
}

impl PathNode {
    /// Return a compact context payload: signature + leading doc/comments only.
    ///
    /// This keeps traversal nodes lightweight and reserves full source for the final target node.
    pub fn skeleton_context(&self) -> String {
        if self.skeleton.is_empty() {
            return build_skeleton_context(&self.body);
        }
        self.skeleton.clone()
    }
}
