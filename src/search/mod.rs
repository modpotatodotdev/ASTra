mod astar;
mod types;

pub use types::{ExecutionPath, PathNode};

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use petgraph::graph::NodeIndex;
use petgraph::Direction;

use crate::embeddings::Embedder;
use crate::graph::CallGraph;
use crate::indexer;
use crate::vector_store::VectorStore;

use astar::{
    AStarNode, AStarOutcome, TELEPORT_MAX_LOCAL_VISITED, TELEPORT_MIN_GAIN, TELEPORT_TOP_K,
    TELEPORT_TRIGGER_MIN_SIMILARITY,
};

/// The search engine combining the call graph and vector store.
pub struct SearchEngine {
    pub graph: CallGraph,
    pub vector_store: VectorStore,
    pub embedder: Box<dyn Embedder>,
}

impl SearchEngine {
    pub fn new(graph: CallGraph, vector_store: VectorStore, embedder: Box<dyn Embedder>) -> Self {
        Self {
            graph,
            vector_store,
            embedder,
        }
    }

    /// Perform semantic path search using bi-directional A* biased by vector similarity.
    ///
    /// 1. Embed the query.
    /// 2. Find top-k entry points via vector similarity.
    /// 3. For each entry point, run bi-directional A* through the call graph.
    /// 4. The A* heuristic is biased by vector similarity to the query.
    /// 5. Return the best execution paths.
    pub fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<ExecutionPath>> {
        let query_embedding = self.embedder.embed(query)?;
        let teleport_candidates = self.precompute_teleport_candidates(&query_embedding);
        let mut similarity_cache: HashMap<NodeIndex, f32> = HashMap::new();

        // Find entry points: top symbols most similar to the query
        let entry_points = self.vector_store.search(&query_embedding, max_results * 2);

        if entry_points.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_paths = Vec::new();

        for entry in &entry_points {
            if let Some(&start_idx) = self.graph.id_to_index.get(&entry.id) {
                // Run bi-directional A* from this entry point
                let paths = self.bidirectional_astar(
                    start_idx,
                    &query_embedding,
                    10,
                    &teleport_candidates,
                    &mut similarity_cache,
                );
                all_paths.extend(paths);
            }
        }

        // Sort by score and deduplicate
        all_paths.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        all_paths.truncate(max_results);
        Ok(all_paths)
    }

    /// Perform regular semantic RAG over indexed symbol chunks (no graph traversal).
    pub fn search_chunks(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<PathNode>> {
        let query_embedding = self.embedder.embed(query)?;
        Ok(self
            .vector_store
            .search(&query_embedding, max_results)
            .into_iter()
            .filter_map(|result| {
                self.graph.id_to_index.get(&result.id).map(|&idx| {
                    let node = &self.graph.graph[idx];
                    PathNode {
                        symbol_id: node.symbol_id.clone(),
                        name: node.name.clone(),
                        symbol_kind: node.kind,
                        file_path: node.file_path.clone(),
                        line_range: node.line_range,
                        body: node.body.clone(),
                        skeleton: node.skeleton.clone(),
                        relevance: result.score,
                    }
                })
            })
            .collect())
    }

    pub fn update_files(
        &mut self,
        config: &crate::config::AstraConfig,
        embedder: &dyn Embedder,
        changed_files: &[String],
    ) -> anyhow::Result<usize> {
        indexer::update_files(
            config,
            &mut self.graph,
            &mut self.vector_store,
            embedder,
            changed_files,
        )
    }

    /// Bi-directional A* search from a start node.
    ///
    /// Forward direction follows callees (outgoing edges).
    /// Backward direction follows callers (incoming edges).
    /// The heuristic biases toward nodes similar to the query embedding.
    fn bidirectional_astar(
        &self,
        start: NodeIndex,
        query_embedding: &[f32],
        max_depth: usize,
        teleport_candidates: &[(NodeIndex, f32)],
        similarity_cache: &mut HashMap<NodeIndex, f32>,
    ) -> Vec<ExecutionPath> {
        let mut paths = Vec::new();

        // Forward search: follow callees
        let forward_path = self.astar_search(
            start,
            query_embedding,
            Direction::Outgoing,
            max_depth,
            teleport_candidates,
            similarity_cache,
        );

        // Backward search: follow callers
        let backward_path = self.astar_search(
            start,
            query_embedding,
            Direction::Incoming,
            max_depth,
            teleport_candidates,
            similarity_cache,
        );

        // Combine backward (reversed) + start + forward into a full path
        let mut combined_nodes: Vec<(NodeIndex, f32)> = Vec::new();

        // Add backward path (reversed, excluding start which is added separately)
        if backward_path.len() > 1 {
            for &(idx, score) in backward_path[1..].iter().rev() {
                combined_nodes.push((idx, score));
            }
        }

        // Add start node
        let start_sim = self
            .vector_store
            .similarity(query_embedding, &self.graph.graph[start].symbol_id);
        combined_nodes.push((start, start_sim));

        // Add forward path (excluding start)
        if forward_path.len() > 1 {
            for &(idx, score) in &forward_path[1..] {
                combined_nodes.push((idx, score));
            }
        }

        // Deduplicate by node index
        let mut seen = HashSet::new();
        combined_nodes.retain(|(idx, _)| seen.insert(*idx));

        if !combined_nodes.is_empty() {
            let total_score: f32 =
                combined_nodes.iter().map(|(_, s)| s).sum::<f32>() / combined_nodes.len() as f32;

            let nodes: Vec<PathNode> = combined_nodes
                .iter()
                .map(|(idx, relevance)| {
                    let node = &self.graph.graph[*idx];
                    PathNode {
                        symbol_id: node.symbol_id.clone(),
                        name: node.name.clone(),
                        symbol_kind: node.kind,
                        file_path: node.file_path.clone(),
                        line_range: node.line_range,
                        body: node.body.clone(),
                        skeleton: node.skeleton.clone(),
                        relevance: *relevance,
                    }
                })
                .collect();

            paths.push(ExecutionPath {
                nodes,
                score: total_score,
            });
        }

        paths
    }

    /// Single-direction A* search through the call graph.
    fn astar_search(
        &self,
        start: NodeIndex,
        query_embedding: &[f32],
        direction: Direction,
        max_depth: usize,
        teleport_candidates: &[(NodeIndex, f32)],
        similarity_cache: &mut HashMap<NodeIndex, f32>,
    ) -> Vec<(NodeIndex, f32)> {
        let primary = self.astar_search_core(
            start,
            query_embedding,
            direction,
            max_depth,
            similarity_cache,
        );
        let should_teleport = (primary.path.len() <= 1
            || primary.best_similarity < TELEPORT_TRIGGER_MIN_SIMILARITY)
            && primary.visited.len() <= TELEPORT_MAX_LOCAL_VISITED;

        if !should_teleport {
            return primary.path;
        }

        let Some(teleport_start) = self.select_teleport_node(
            teleport_candidates,
            &primary.visited,
            primary.best_similarity,
        ) else {
            return primary.path;
        };

        let teleported = self.astar_search_core(
            teleport_start,
            query_embedding,
            direction,
            max_depth,
            similarity_cache,
        );
        if teleported.path.is_empty() {
            return primary.path;
        }

        let mut combined = primary.path;
        for (idx, sim) in teleported.path {
            if !combined.iter().any(|(existing, _)| *existing == idx) {
                combined.push((idx, sim));
            }
        }
        combined
    }

    fn astar_search_core(
        &self,
        start: NodeIndex,
        query_embedding: &[f32],
        direction: Direction,
        max_depth: usize,
        similarity_cache: &mut HashMap<NodeIndex, f32>,
    ) -> AStarOutcome {
        let mut open = BinaryHeap::new();
        let mut came_from: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut g_scores: HashMap<NodeIndex, f32> = HashMap::new();
        let mut visited = HashSet::new();

        g_scores.insert(start, 0.0);
        let h_start = self.node_similarity_cached(start, query_embedding, similarity_cache);
        open.push(AStarNode {
            index: start,
            g_score: 0.0,
            f_score: 1.0 - h_start,
        });

        let mut best_node = start;
        let mut best_similarity = h_start;
        let mut depth_map: HashMap<NodeIndex, usize> = HashMap::new();
        depth_map.insert(start, 0);

        while let Some(current) = open.pop() {
            let current_idx = current.index;

            if visited.contains(&current_idx) {
                continue;
            }
            visited.insert(current_idx);

            let current_depth = depth_map.get(&current_idx).copied().unwrap_or(0);
            if current_depth >= max_depth {
                continue;
            }

            // Track the best node found so far
            let sim = self.node_similarity_cached(current_idx, query_embedding, similarity_cache);
            if sim > best_similarity {
                best_similarity = sim;
                best_node = current_idx;
            }

            // Expand neighbors
            let neighbors: Vec<NodeIndex> = self
                .graph
                .graph
                .neighbors_directed(current_idx, direction)
                .collect();

            for next in neighbors {
                if visited.contains(&next) {
                    continue;
                }

                let depth_penalty = 0.15;
                let tentative_g = current.g_score + depth_penalty;
                let current_g = g_scores.get(&next).copied().unwrap_or(f32::INFINITY);

                if tentative_g < current_g {
                    came_from.insert(next, current_idx);
                    g_scores.insert(next, tentative_g);
                    depth_map.insert(next, current_depth + 1);

                    let h = self.node_similarity_cached(next, query_embedding, similarity_cache);
                    open.push(AStarNode {
                        index: next,
                        g_score: tentative_g,
                        // Lower f_score = higher priority; higher similarity = lower cost
                        f_score: tentative_g + (1.0 - h),
                    });
                }
            }
        }

        // Reconstruct path from start to best node
        let mut path = Vec::new();
        let mut current = best_node;
        loop {
            let sim = self.node_similarity_cached(current, query_embedding, similarity_cache);
            path.push((current, sim));
            if current == start {
                break;
            }
            match came_from.get(&current) {
                Some(&prev) => current = prev,
                None => break,
            }
        }
        path.reverse();
        AStarOutcome {
            path,
            best_similarity,
            visited,
        }
    }

    fn select_teleport_node(
        &self,
        teleport_candidates: &[(NodeIndex, f32)],
        visited: &HashSet<NodeIndex>,
        best_similarity: f32,
    ) -> Option<NodeIndex> {
        for (idx, score) in teleport_candidates {
            if *score <= best_similarity + TELEPORT_MIN_GAIN {
                continue;
            }
            if visited.contains(idx) {
                continue;
            }
            return Some(*idx);
        }
        None
    }

    fn precompute_teleport_candidates(&self, query_embedding: &[f32]) -> Vec<(NodeIndex, f32)> {
        self.vector_store
            .search(query_embedding, TELEPORT_TOP_K)
            .into_iter()
            .filter_map(|candidate| {
                self.graph
                    .id_to_index
                    .get(&candidate.id)
                    .copied()
                    .map(|idx| (idx, candidate.score))
            })
            .collect()
    }

    fn node_similarity_cached(
        &self,
        node: NodeIndex,
        query_embedding: &[f32],
        similarity_cache: &mut HashMap<NodeIndex, f32>,
    ) -> f32 {
        if let Some(score) = similarity_cache.get(&node) {
            return *score;
        }
        let score = self.node_similarity(node, query_embedding);
        similarity_cache.insert(node, score);
        score
    }

    /// Heuristic function: vector similarity between a node and the query.
    /// Get the vector similarity between a graph node and a query embedding.
    fn node_similarity(&self, node: NodeIndex, query_embedding: &[f32]) -> f32 {
        let symbol_id = &self.graph.graph[node].symbol_id;
        self.vector_store.similarity(query_embedding, symbol_id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use petgraph::Direction;

    use super::{PathNode, SearchEngine};
    use crate::embeddings::{Embedder, SemanticEmbedder, DEFAULT_LOCAL_DIM};
    use crate::graph::CallGraph;
    use crate::parser::{parse_source, Language, SymbolKind};
    use crate::test_helpers::build_test_engine_from_source;
    use crate::vector_store::VectorStore;

    const SOURCE: &str = r#"
fn router() {
    auth_middleware();
}

fn auth_middleware() {
    db_insert();
}

fn db_insert() {
    println!("inserting into database");
}

fn unrelated_math() {
    let x = 42 + 8;
}
"#;

    fn build_test_engine() -> SearchEngine {
        build_test_engine_from_source(SOURCE, "main.rs")
    }

    #[test]
    fn test_search_finds_relevant_path() {
        let engine = build_test_engine();
        let results = engine.search("database insert", 3).unwrap();

        assert!(!results.is_empty(), "should find at least one path");

        let all_names: Vec<&str> = results
            .iter()
            .flat_map(|p| p.nodes.iter().map(|n| n.name.as_str()))
            .collect();
        assert!(
            all_names.contains(&"db_insert"),
            "search for 'database insert' should find db_insert, got: {:?}",
            all_names
        );
    }

    #[test]
    fn test_search_returns_execution_path() {
        let engine = build_test_engine();
        let results = engine.search("auth middleware routing", 1).unwrap();

        assert!(!results.is_empty());
        let path = &results[0];
        assert!(path.nodes.len() >= 1, "path should have at least one node");
    }

    #[test]
    fn test_empty_search() {
        let graph = CallGraph::new();
        let vector_store = VectorStore::new(DEFAULT_LOCAL_DIM);
        let embedder = SemanticEmbedder::try_new().unwrap();
        let engine = SearchEngine::new(graph, vector_store, Box::new(embedder));

        let results = engine.search("anything", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_chunk_search_finds_relevant_symbol() {
        let engine = build_test_engine();
        let results = engine.search_chunks("database insert", 3).unwrap();

        assert!(!results.is_empty(), "should find at least one symbol chunk");
        let names: Vec<&str> = results.iter().map(|n| n.name.as_str()).collect();
        assert!(
            names.contains(&"db_insert"),
            "chunk search for 'database insert' should find db_insert, got: {:?}",
            names
        );
    }

    #[test]
    fn test_skeleton_context_signature_and_docstring_only() {
        let node = PathNode {
            symbol_id: "main.rs::compute_total".to_string(),
            name: "compute_total".to_string(),
            symbol_kind: SymbolKind::Function,
            file_path: "main.rs".to_string(),
            line_range: (0, 10),
            body: r#"def compute_total(items, tax_rate):
    """Return total amount including tax."""
    subtotal = sum(items)
    return subtotal * (1 + tax_rate)
"#
            .to_string(),
            skeleton: "def compute_total(items, tax_rate):\n\n    \"\"\"Return total amount including tax.\"\"\"".to_string(),
            relevance: 0.9,
        };

        let skeleton = node.skeleton_context();
        assert!(skeleton.contains("def compute_total(items, tax_rate):"));
        assert!(skeleton.contains("Return total amount including tax"));
        assert!(!skeleton.contains("subtotal = sum(items)"));
    }

    #[test]
    fn test_astar_teleport_can_reach_disconnected_high_similarity_node() {
        let source = r#"
fn router_entry() {
    auth_gate();
}

fn auth_gate() {
    let token = "x";
}

fn db_insert_payload() {
    println!("database insert write path");
}
"#;
        let parsed = parse_source(source, Language::Rust, "main.rs").unwrap();
        let mut graph = CallGraph::new();
        graph.build_from_files(&[parsed]);

        let embedder = SemanticEmbedder::try_new().unwrap();
        let mut vector_store = VectorStore::new(DEFAULT_LOCAL_DIM);
        for idx in graph.all_node_indices() {
            let node = &graph.graph[idx];
            let embedding = embedder.embed(&node.body).unwrap();
            vector_store.upsert(&node.symbol_id, embedding);
        }

        let engine = SearchEngine::new(graph, vector_store, Box::new(embedder));
        let start = engine.graph.id_to_index["main.rs::router_entry"];
        let query_embedding = engine.embedder.embed("database insert write bug").unwrap();
        let teleport_candidates = engine.precompute_teleport_candidates(&query_embedding);
        let mut similarity_cache = HashMap::new();

        let path = engine.astar_search(
            start,
            &query_embedding,
            Direction::Outgoing,
            1,
            &teleport_candidates,
            &mut similarity_cache,
        );
        let names: Vec<&str> = path
            .iter()
            .map(|(idx, _)| engine.graph.graph[*idx].name.as_str())
            .collect();

        assert!(
            names.contains(&"db_insert_payload"),
            "teleport fallback should include disconnected relevant node, got: {:?}",
            names
        );
    }
}
