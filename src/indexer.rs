use std::path::Path;
use std::time::SystemTime;
use std::time::{Duration, Instant};

use anyhow::Result;
use log::info;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::config::AstraConfig;
use crate::embeddings::Embedder;
use crate::graph::CallGraph;
use crate::parser::{self, Language, ParsedFile};
use crate::storage;
use crate::vector_store::VectorStore;

/// Result of an indexing run.
pub struct IndexResult {
    pub graph: CallGraph,
    pub vector_store: VectorStore,
    pub files_indexed: usize,
    pub symbols_indexed: usize,
}

/// Validate that graph and vector-store are internally consistent.
///
/// Returns true when each graph node has a corresponding vector and dimensions
/// match the vector store expectations.
pub fn validate_index_integrity(
    graph: &CallGraph,
    vector_store: &VectorStore,
    expected_dim: usize,
) -> bool {
    if vector_store.dim != expected_dim {
        return false;
    }

    for idx in graph.all_node_indices() {
        let node = &graph.graph[idx];
        if !vector_store.contains_id(&node.symbol_id) {
            return false;
        }
    }
    true
}

/// Perform cold-start indexing of the entire workspace using rayon parallelism.
pub fn index_workspace(config: &AstraConfig, embedder: &dyn Embedder) -> Result<IndexResult> {
    info!(
        "Starting cold-start indexing of {}",
        config.workspace_root.display()
    );

    // Collect all indexable files
    let files = collect_files(&config.workspace_root, &config.extensions)?;
    info!("Found {} files to index", files.len());

    // Parse all files in parallel using rayon
    let parsed_files: Vec<ParsedFile> = files
        .par_iter()
        .filter_map(
            |path| match parser::parse_file(Path::new(path), &config.workspace_root) {
                Ok(parsed) => Some(parsed),
                Err(e) => {
                    log::warn!("Failed to parse {}: {}", path, e);
                    None
                }
            },
        )
        .collect();

    // Build call graph
    let mut graph = CallGraph::new();
    graph.build_from_files(&parsed_files);
    info!(
        "Built call graph: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Build vector store: embed all symbols in batches
    let mut vector_store = VectorStore::new(embedder.dim());

    let node_indices = graph.all_node_indices();
    let chunks: Vec<Vec<_>> = node_indices.chunks(64).map(|c| c.to_vec()).collect();
    let mut indexed = 0usize;
    let mut last_progress_log = Instant::now();
    for chunk in chunks {
        let texts: Vec<&str> = chunk
            .iter()
            .map(|&idx| graph.graph[idx].body.as_str())
            .collect();
        let expected_embeddings = texts.len();
        let batch_embeddings = match embedder.embed_batch(texts) {
            Ok(v) => v,
            Err(e) => {
                log::error!("Embedding batch failed, skipping chunk: {}", e);
                continue;
            }
        };
        if batch_embeddings.len() != expected_embeddings {
            log::warn!(
                "Skipping chunk due to embedding batch mismatch: expected {}, got {}",
                expected_embeddings,
                batch_embeddings.len()
            );
            continue;
        }

        for (idx, emb) in chunk.into_iter().zip(batch_embeddings) {
            if emb.len() != embedder.dim() {
                log::warn!(
                    "Skipping symbol {} due to embedding size mismatch: expected {}, got {}",
                    graph.graph[idx].symbol_id,
                    embedder.dim(),
                    emb.len()
                );
                continue;
            }
            vector_store.upsert(&graph.graph[idx].symbol_id, emb);
            indexed += 1;
        }

        let should_log =
            last_progress_log.elapsed() >= Duration::from_secs(1) || indexed == node_indices.len();
        if should_log {
            info!("Indexing {}/{}", indexed, node_indices.len());
            last_progress_log = Instant::now();
        }
    }

    let symbols_indexed = node_indices.len();
    info!("Indexed {} symbols into vector store", symbols_indexed);

    // Save to disk
    storage::save_graph(config, &graph)?;
    storage::save_vector_store(config, &vector_store)?;

    // Save metadata with file timestamps
    let mut metadata = storage::IndexMetadata::default();
    for path in &files {
        if let Ok(ts) = file_modified_timestamp(path) {
            let relative = Path::new(path)
                .strip_prefix(&config.workspace_root)
                .unwrap_or(Path::new(path))
                .to_string_lossy()
                .to_string();
            metadata.file_timestamps.insert(relative, ts);
        }
    }
    storage::save_metadata(config, &metadata)?;

    Ok(IndexResult {
        graph,
        vector_store,
        files_indexed: files.len(),
        symbols_indexed,
    })
}

/// Incrementally update the index for changed files.
pub fn update_files(
    config: &AstraConfig,
    graph: &mut CallGraph,
    vector_store: &mut VectorStore,
    embedder: &dyn Embedder,
    changed_files: &[String],
) -> Result<usize> {
    let mut updated_count = 0;

    for file_path in changed_files {
        let abs_path = config.workspace_root.join(file_path);

        // Prevent path traversal: reject paths that could escape the workspace.
        // First, check the raw joined path for obvious traversal before any I/O.
        if !abs_path.starts_with(&config.workspace_root) {
            log::warn!("Skipping path outside workspace: {}", file_path);
            continue;
        }
        // Second, if the file exists, verify the canonical (symlink-resolved) path
        // is still within the workspace.
        if abs_path.exists() {
            match (
                abs_path.canonicalize(),
                config.workspace_root.canonicalize(),
            ) {
                (Ok(canonical_path), Ok(canonical_root)) => {
                    if !canonical_path.starts_with(&canonical_root) {
                        log::warn!("Skipping symlink escaping workspace: {}", file_path);
                        continue;
                    }
                }
                _ => {
                    log::warn!(
                        "Skipping path that could not be canonicalized: {}",
                        file_path
                    );
                    continue;
                }
            }
        }

        // Remove old data for this file
        graph.remove_file(file_path);
        vector_store.remove_by_prefix(&format!("{}::", file_path));

        // Re-parse if file still exists
        if abs_path.exists() {
            match parser::parse_file(&abs_path, &config.workspace_root) {
                Ok(parsed) => {
                    graph.build_from_files(&[parsed]);

                    // Re-embed all symbols from this file
                    let mut nodes_to_embed = Vec::new();
                    for idx in graph.all_node_indices() {
                        let node = &graph.graph[idx];
                        if node.file_path == *file_path {
                            nodes_to_embed.push(node);
                        }
                    }

                    if !nodes_to_embed.is_empty() {
                        for chunk in nodes_to_embed.chunks(64) {
                            let texts: Vec<&str> = chunk.iter().map(|n| n.body.as_str()).collect();
                            let expected_embeddings = texts.len();
                            let batch_embeddings = match embedder.embed_batch(texts) {
                                Ok(v) => v,
                                Err(e) => {
                                    log::error!(
                                        "Embedding batch failed during update, skipping chunk: {}",
                                        e
                                    );
                                    continue;
                                }
                            };
                            if batch_embeddings.len() != expected_embeddings {
                                log::warn!(
                                    "Skipping update chunk due to embedding batch mismatch: expected {}, got {}",
                                    expected_embeddings,
                                    batch_embeddings.len()
                                );
                                continue;
                            }
                            for (node, embedding) in chunk.iter().zip(batch_embeddings) {
                                if embedding.len() != embedder.dim() {
                                    log::warn!(
                                        "Skipping symbol {} update due to embedding size mismatch: expected {}, got {}",
                                        node.symbol_id,
                                        embedder.dim(),
                                        embedding.len()
                                    );
                                    continue;
                                }
                                vector_store.upsert(&node.symbol_id, embedding);
                                updated_count += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to re-parse {}: {}", file_path, e);
                }
            }
        }
    }

    // Persist updated data
    storage::save_graph(config, graph)?;
    storage::save_vector_store(config, vector_store)?;

    Ok(updated_count)
}

/// Collect all files matching the configured extensions.
pub fn collect_files(root: &Path, extensions: &[String]) -> Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_entry(|e| {
        // Always include the root directory itself
        if e.depth() == 0 {
            return true;
        }
        let name = e.file_name().to_string_lossy();
        // Skip hidden dirs, target, node_modules, etc.
        !name.starts_with('.')
            && name != "target"
            && name != "node_modules"
            && name != "__pycache__"
            && name != "vendor"
    }) {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                if extensions.iter().any(|x| x == ext) {
                    // Also check if the language is supported
                    if Language::from_extension(ext).is_some() {
                        files.push(entry.path().to_string_lossy().to_string());
                    }
                }
            }
        }
    }
    Ok(files)
}

/// Get the last-modified timestamp of a file as seconds since UNIX epoch.
fn file_modified_timestamp(path: &str) -> Result<u64> {
    let metadata = std::fs::metadata(path)?;
    let modified = metadata.modified()?;
    Ok(modified
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::{collect_files, index_workspace, update_files};
    use crate::config::AstraConfig;
    use crate::embeddings::{Embedder, SemanticEmbedder, DEFAULT_LOCAL_DIM};

    #[test]
    fn test_index_workspace() {
        let tmp = TempDir::new().unwrap();
        let src_dir = tmp.path().join("src");
        fs::create_dir_all(&src_dir).unwrap();

        fs::write(
            src_dir.join("main.rs"),
            r#"
fn main() {
    greet("world");
}

fn greet(name: &str) {
    println!("Hello, {}!", name);
}
"#,
        )
        .unwrap();

        let config = AstraConfig::new(tmp.path());
        let embedder = SemanticEmbedder::try_new().unwrap();
        let result = index_workspace(&config, &embedder).unwrap();

        assert_eq!(result.files_indexed, 1);
        assert_eq!(result.symbols_indexed, 2);
        assert_eq!(result.graph.node_count(), 2);

        // Verify persistence
        assert!(config.graph_path().exists());
        assert!(config.vector_db_path().exists());
    }

    #[test]
    fn test_incremental_update() {
        let tmp = TempDir::new().unwrap();

        fs::write(
            tmp.path().join("lib.rs"),
            "fn old_function() { println!(\"old\"); }\n",
        )
        .unwrap();

        let config = AstraConfig::new(tmp.path());
        let embedder = SemanticEmbedder::try_new().unwrap();
        let mut result = index_workspace(&config, &embedder).unwrap();

        assert_eq!(result.graph.node_count(), 1);

        // Update the file
        fs::write(
            tmp.path().join("lib.rs"),
            "fn new_function() { println!(\"new\"); }\nfn another() {}\n",
        )
        .unwrap();

        let updated = update_files(
            &config,
            &mut result.graph,
            &mut result.vector_store,
            &embedder,
            &["lib.rs".to_string()],
        )
        .unwrap();

        assert_eq!(updated, 2);
        assert_eq!(result.graph.node_count(), 2);
    }

    #[test]
    fn test_collect_files_skips_hidden() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("visible.rs"), "fn v() {}").unwrap();
        let hidden = tmp.path().join(".hidden");
        fs::create_dir_all(&hidden).unwrap();
        fs::write(hidden.join("secret.rs"), "fn s() {}").unwrap();

        let files = collect_files(tmp.path(), &["rs".into()]).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].contains("visible.rs"));
    }

    #[test]
    fn test_update_files_rejects_path_traversal() {
        let tmp = TempDir::new().unwrap();

        fs::write(
            tmp.path().join("lib.rs"),
            "fn original() { println!(\"orig\"); }\n",
        )
        .unwrap();

        let config = AstraConfig::new(tmp.path());
        let embedder = SemanticEmbedder::try_new().unwrap();
        let mut result = index_workspace(&config, &embedder).unwrap();

        // Attempt path traversal — should be silently skipped
        let updated = update_files(
            &config,
            &mut result.graph,
            &mut result.vector_store,
            &embedder,
            &["../../etc/passwd".to_string()],
        )
        .unwrap();

        assert_eq!(updated, 0, "path traversal should be rejected");
    }

    #[test]
    fn test_validate_index_integrity_ok_and_missing_vector() {
        let tmp = TempDir::new().unwrap();

        fs::write(tmp.path().join("lib.rs"), "fn a() {}\nfn b() { a(); }\n").unwrap();

        let config = AstraConfig::new(tmp.path());
        let embedder = SemanticEmbedder::try_new().unwrap();
        let result = index_workspace(&config, &embedder).unwrap();

        assert!(super::validate_index_integrity(
            &result.graph,
            &result.vector_store,
            embedder.dim()
        ));

        let mut broken_store = result.vector_store.clone();
        broken_store.remove("lib.rs::a");

        assert!(!super::validate_index_integrity(
            &result.graph,
            &broken_store,
            embedder.dim()
        ));
    }

    #[test]
    fn test_validate_index_integrity_rejects_dimension_mismatch() {
        let mut graph = crate::graph::CallGraph::new();
        let parsed =
            crate::parser::parse_source("fn f() {}", crate::parser::Language::Rust, "f.rs")
                .unwrap();
        graph.build_from_files(&[parsed]);

        let store = crate::vector_store::VectorStore::new(123);
        assert!(!super::validate_index_integrity(
            &graph,
            &store,
            DEFAULT_LOCAL_DIM
        ));
    }
}
