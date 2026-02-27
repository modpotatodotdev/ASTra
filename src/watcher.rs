use std::time::Duration;

use anyhow::Result;
use log::info;
use notify::RecursiveMode;
use notify_debouncer_mini::new_debouncer;
use tokio::sync::mpsc;

use crate::config::AstraConfig;
use crate::embeddings::Embedder;
use crate::graph::CallGraph;
use crate::indexer;
use crate::parser::Language;
use crate::vector_store::VectorStore;

/// File change event for the indexer.
#[derive(Debug, Clone)]
pub struct FileChange {
    pub path: String,
    pub kind: ChangeKind,
}

#[derive(Debug, Clone)]
pub enum ChangeKind {
    Modified,
    Created,
    Deleted,
}

/// Start watching the workspace for file changes.
/// Returns a channel receiver that emits batches of changed file paths.
pub async fn watch_workspace(config: &AstraConfig) -> Result<mpsc::Receiver<Vec<String>>> {
    let (tx, rx) = mpsc::channel::<Vec<String>>(100);
    let workspace_root = config.workspace_root.clone();
    let extensions: Vec<String> = config.extensions.clone();

    tokio::task::spawn_blocking(move || {
        let (notify_tx, notify_rx) = std::sync::mpsc::channel();

        let mut debouncer = match new_debouncer(Duration::from_millis(500), notify_tx) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Failed to create file watcher: {}", e);
                return;
            }
        };

        if let Err(e) = debouncer
            .watcher()
            .watch(&workspace_root, RecursiveMode::Recursive)
        {
            log::error!("Failed to watch workspace: {}", e);
            return;
        }

        info!("Watching {} for changes", workspace_root.display());

        loop {
            match notify_rx.recv() {
                Ok(Ok(events)) => {
                    let mut changed_files = Vec::new();
                    for event in events {
                        let path = &event.path;

                        // Skip hidden files and directories
                        if path
                            .components()
                            .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
                        {
                            continue;
                        }

                        // Skip non-matching extensions
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            if extensions.iter().any(|x| x == ext)
                                && Language::from_extension(ext).is_some()
                            {
                                let relative = path
                                    .strip_prefix(&workspace_root)
                                    .unwrap_or(path)
                                    .to_string_lossy()
                                    .to_string();
                                changed_files.push(relative);
                            }
                        }
                    }

                    if !changed_files.is_empty() {
                        changed_files.sort();
                        changed_files.dedup();
                        info!("Detected changes in {} files", changed_files.len());
                        if tx.blocking_send(changed_files).is_err() {
                            break;
                        }
                    }
                }
                Ok(Err(errors)) => {
                    log::warn!("Watch error: {:?}", errors);
                }
                Err(_) => break,
            }
        }
    });

    Ok(rx)
}

/// Process incoming file changes and update the index.
pub async fn process_changes(
    config: &AstraConfig,
    graph: &mut CallGraph,
    vector_store: &mut VectorStore,
    embedder: &dyn Embedder,
    rx: &mut mpsc::Receiver<Vec<String>>,
) -> Result<()> {
    while let Some(changed_files) = rx.recv().await {
        info!("Processing {} changed files", changed_files.len());
        let updated = indexer::update_files(config, graph, vector_store, embedder, &changed_files)?;
        info!("Updated {} symbols", updated);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ChangeKind, FileChange};

    #[test]
    fn test_file_change_struct() {
        let change = FileChange {
            path: "src/main.rs".to_string(),
            kind: ChangeKind::Modified,
        };
        assert_eq!(change.path, "src/main.rs");
    }
}
