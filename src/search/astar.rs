use std::cmp::Ordering;
use std::collections::HashSet;

use petgraph::graph::NodeIndex;

/// A* node state for the priority queue.
#[derive(Debug, Clone)]
pub(super) struct AStarNode {
    pub index: NodeIndex,
    /// g(n): cost from start to this node.
    pub g_score: f32,
    /// f(n) = g(n) + h(n): estimated total cost.
    pub f_score: f32,
}

#[derive(Debug, Clone)]
pub(super) struct AStarOutcome {
    pub path: Vec<(NodeIndex, f32)>,
    pub best_similarity: f32,
    pub visited: HashSet<NodeIndex>,
}

pub(super) const TELEPORT_TOP_K: usize = 32;
pub(super) const TELEPORT_TRIGGER_MIN_SIMILARITY: f32 = 0.55;
pub(super) const TELEPORT_MIN_GAIN: f32 = 0.03;
pub(super) const TELEPORT_MAX_LOCAL_VISITED: usize = 24;

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score
    }
}
impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lower f_score = higher priority)
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}
