use std::env;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use log::info;

use astra::config::AstraConfig;

use astra::indexer;
use astra::mcp::McpServer;
use astra::search::SearchEngine;
use astra::storage;
use astra::watcher;

fn main() -> Result<()> {
    env_logger::init();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let workspace_root = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().expect("failed to get current directory"));

    let workspace_root = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.clone());

    info!("ASTra starting for workspace: {}", workspace_root.display());

    let config = AstraConfig::new(&workspace_root);

    let embedder = astra::embeddings::build_embedder(config.embedding_provider.as_str())?;

    // Cold-start index or load from disk.
    // On format migration (e.g. f32→f16 vectors, zstd graph compression) the
    // deserializer will return an error; we catch it here and reindex rather
    // than crashing the MCP server.
    let (graph, vector_store) = if storage::has_persisted_data(&config) {
        info!("Loading persisted index from {}", config.data_dir.display());
        let load_result: anyhow::Result<_> = (|| {
            let mut g = storage::load_graph(&config)?;
            g.rebuild_after_deserialize();
            let mut v = storage::load_vector_store(&config)?;
            v.rebuild_index();
            anyhow::Ok((g, v))
        })();

        match load_result {
            Ok((graph, vector_store))
                if indexer::validate_index_integrity(&graph, &vector_store, embedder.dim()) =>
            {
                info!(
                    "Loaded graph ({} nodes, {} edges) and vector store ({} vectors)",
                    graph.node_count(),
                    graph.edge_count(),
                    vector_store.len()
                );
                (graph, vector_store)
            }
            other => {
                if other.is_err() {
                    info!("Persisted index could not be loaded (format migration?); rebuilding...");
                } else {
                    info!("Persisted index integrity check failed; rebuilding...");
                }
                let result = indexer::index_workspace(&config, embedder.as_ref())?;
                info!(
                    "Indexed {} files, {} symbols",
                    result.files_indexed, result.symbols_indexed
                );
                (result.graph, result.vector_store)
            }
        }
    } else {
        info!("No persisted data found, performing cold-start indexing...");
        let result = indexer::index_workspace(&config, embedder.as_ref())?;
        info!(
            "Indexed {} files, {} symbols",
            result.files_indexed, result.symbols_indexed
        );
        (result.graph, result.vector_store)
    };

    // Create search engine
    let engine = Arc::new(RwLock::new(SearchEngine::new(
        graph,
        vector_store,
        embedder,
    )));

    let config_for_watcher = config.clone();
    let engine_for_watcher = Arc::clone(&engine);
    rt.spawn(async move {
        let watcher_embedder = match astra::embeddings::build_embedder(config_for_watcher.embedding_provider.as_str()) {
            Ok(embedder) => embedder,
            Err(e) => {
                log::warn!(
                    "File watcher disabled: failed to initialize embedder: {}",
                    e
                );
                return;
            }
        };
        match watcher::watch_workspace(&config_for_watcher).await {
            Ok(mut rx) => {
                while let Some(changed_files) = rx.recv().await {
                    if changed_files.is_empty() {
                        continue;
                    }
                    let mut guard = match engine_for_watcher.write() {
                        Ok(g) => g,
                        Err(e) => {
                            log::error!("Failed to lock search engine for watcher updates: {}", e);
                            break;
                        }
                    };
                    let updated =
                        guard.update_files(&config_for_watcher, watcher_embedder.as_ref(), &changed_files);
                    match updated {
                        Ok(count) => info!("Watcher updated {} symbols", count),
                        Err(e) => log::warn!("Watcher failed to update files: {}", e),
                    }
                }
            }
            Err(e) => log::warn!("File watcher disabled: {}", e),
        }
    });

    // Start MCP server over stdio
    let server = McpServer::new_shared(engine);
    server.run()?;

    Ok(())
}



