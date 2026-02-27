use std::collections::HashMap;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::embeddings::cosine_similarity_f16;

/// A fast in-memory vector store for similarity search.
///
/// Stores embedding vectors keyed by symbol ID and supports nearest-neighbor
/// queries via brute-force cosine similarity. For codebases up to ~100k symbols,
/// brute-force is fast enough (sub-millisecond on modern CPUs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStore {
    /// Embedding dimensionality.
    pub dim: usize,
    /// Stored entries: (symbol_id, embedding vector).
    entries: Vec<VectorEntry>,
    /// Index from symbol ID to position in `entries` for O(1) lookups.
    #[serde(skip)]
    id_index: HashMap<String, usize>,
}

/// Embedding stored as f16 to halve memory and disk usage.
/// Quantized once on insert; up-converted element-by-element at query time.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    id: String,
    vector: Vec<f16>,
}

/// A search result with score.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub id: String,
    pub score: f32,
}

impl VectorStore {
    /// Create a new empty vector store with given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
            id_index: HashMap::new(),
        }
    }

    /// Rebuild the id_index after deserialization (since it is `serde(skip)`).
    pub fn rebuild_index(&mut self) {
        self.id_index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.id_index.insert(entry.id.clone(), i);
        }
    }

    /// Insert or update a vector for the given symbol ID.
    /// The f32 vector is quantized to f16 once on insert; no further copies are made.
    pub fn upsert(&mut self, id: &str, vector: Vec<f32>) {
        debug_assert_eq!(vector.len(), self.dim);
        let quantized: Vec<f16> = vector.iter().map(|&x| f16::from_f32(x)).collect();
        if let Some(&pos) = self.id_index.get(id) {
            self.entries[pos].vector = quantized;
        } else {
            let pos = self.entries.len();
            self.entries.push(VectorEntry {
                id: id.to_string(),
                vector: quantized,
            });
            self.id_index.insert(id.to_string(), pos);
        }
    }

    /// Remove all vectors for symbols from a given file path prefix.
    pub fn remove_by_prefix(&mut self, prefix: &str) {
        self.entries.retain(|e| !e.id.starts_with(prefix));
        self.rebuild_index();
    }

    /// Remove a specific vector by ID.
    pub fn remove(&mut self, id: &str) {
        if let Some(pos) = self.id_index.remove(id) {
            self.entries.swap_remove(pos);
            // Update the index for the entry that was swapped into `pos`
            if pos < self.entries.len() {
                self.id_index.insert(self.entries[pos].id.clone(), pos);
            }
        }
    }

    /// Find the top-k most similar vectors to the query.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SimilarityResult> {
        debug_assert_eq!(query.len(), self.dim);
        if top_k == 0 || self.entries.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| (i, cosine_similarity_f16(query, &entry.vector)))
            .collect();
        let score_desc = |a: &(usize, f32), b: &(usize, f32)| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        };

        if top_k > 0 && top_k < scored.len() {
            scored.select_nth_unstable_by(top_k - 1, score_desc);
            scored.truncate(top_k);
        }

        // Sort only the top-k set by score descending.
        scored.sort_unstable_by(score_desc);

        // Only clone IDs for the top-k results
        scored
            .into_iter()
            .map(|(i, score)| SimilarityResult {
                id: self.entries[i].id.clone(),
                score,
            })
            .collect()
    }

    /// Get the similarity score between a query and a specific symbol.
    pub fn similarity(&self, query: &[f32], id: &str) -> f32 {
        self.id_index
            .get(id)
            .map(|&pos| cosine_similarity_f16(query, &self.entries[pos].vector))
            .unwrap_or(0.0)
    }

    /// Check whether an embedding exists for a symbol ID.
    pub fn contains_id(&self, id: &str) -> bool {
        self.id_index.contains_key(id)
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::VectorStore;
    use crate::embeddings::{Embedder, SemanticEmbedder, DEFAULT_LOCAL_DIM};

    #[test]
    fn test_upsert_and_search() {
        let embedder = SemanticEmbedder::try_new().unwrap();
        let mut store = VectorStore::new(DEFAULT_LOCAL_DIM);

        store.upsert("fn_a", embedder.embed("fetch user from database").unwrap());
        store.upsert("fn_b", embedder.embed("calculate tax rate").unwrap());
        store.upsert("fn_c", embedder.embed("get user data from db").unwrap());

        let query = embedder.embed("user database query").unwrap();
        let results = store.search(&query, 2);

        assert_eq!(results.len(), 2);
        let top_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(
            top_ids.contains(&"fn_a") || top_ids.contains(&"fn_c"),
            "expected user-related functions in top results: {:?}",
            top_ids
        );
    }

    #[test]
    fn test_remove() {
        let mut store = VectorStore::new(4);
        store.upsert("a", vec![1.0, 0.0, 0.0, 0.0]);
        store.upsert("b", vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(store.len(), 2);

        store.remove("a");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_remove_by_prefix() {
        let mut store = VectorStore::new(4);
        store.upsert("file1.rs::fn_a", vec![1.0, 0.0, 0.0, 0.0]);
        store.upsert("file1.rs::fn_b", vec![0.0, 1.0, 0.0, 0.0]);
        store.upsert("file2.rs::fn_c", vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(store.len(), 3);

        store.remove_by_prefix("file1.rs");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_search_top_k_zero_returns_empty() {
        let mut store = VectorStore::new(2);
        store.upsert("a", vec![1.0, 0.0]);
        let results = store.search(&[1.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_top_k_matches_full_sort_expectation() {
        let mut store = VectorStore::new(2);
        store.upsert("a", vec![1.0, 0.0]);
        store.upsert("b", vec![0.8, 0.2]);
        store.upsert("c", vec![0.0, 1.0]);
        store.upsert("d", vec![-1.0, 0.0]);

        let query = [1.0, 0.0];
        let top_k = 2;
        let results = store.search(&query, top_k);

        let mut expected: Vec<(&str, f32)> = vec![
            (
                "a",
                crate::embeddings::cosine_similarity(&query, &[1.0, 0.0]),
            ),
            (
                "b",
                crate::embeddings::cosine_similarity(&query, &[0.8, 0.2]),
            ),
            (
                "c",
                crate::embeddings::cosine_similarity(&query, &[0.0, 1.0]),
            ),
            (
                "d",
                crate::embeddings::cosine_similarity(&query, &[-1.0, 0.0]),
            ),
        ];
        expected.sort_by(|lhs, rhs| {
            rhs.1
                .partial_cmp(&lhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        expected.truncate(top_k);

        assert_eq!(results.len(), top_k);
        let expected_ids: Vec<&str> = expected.iter().map(|(id, _)| *id).collect();
        let result_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(result_ids, expected_ids);
    }
}
