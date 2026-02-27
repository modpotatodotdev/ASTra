//! Shared test utilities for building test fixtures.
//!
//! This module is only compiled when running tests (`cfg(test)`).

use crate::embeddings::{Embedder, SemanticEmbedder};
use crate::graph::CallGraph;
use crate::parser::{parse_source, Language};
use crate::search::SearchEngine;
use crate::vector_store::VectorStore;

/// Build a [`SearchEngine`] from Rust source code for testing purposes.
///
/// Parses the provided source, builds a call graph, embeds all symbols,
/// and returns a fully initialised `SearchEngine`.
pub fn build_test_engine_from_source(source: &str, file_name: &str) -> SearchEngine {
    let parsed = parse_source(source, Language::Rust, file_name).unwrap();
    let mut graph = CallGraph::new();
    graph.build_from_files(&[parsed]);

    let embedder = SemanticEmbedder::try_new().unwrap();
    let mut vector_store = VectorStore::new(embedder.dim());

    for idx in graph.all_node_indices() {
        let node = &graph.graph[idx];
        let embedding = embedder.embed(&node.body).unwrap();
        vector_store.upsert(&node.symbol_id, embedding);
    }

    SearchEngine::new(graph, vector_store, Box::new(embedder))
}
