use std::path::{Path, PathBuf};

/// Configuration for the ASTRA indexing and search engine.
#[derive(Debug, Clone)]
pub struct AstraConfig {
    /// Root of the workspace being indexed.
    pub workspace_root: PathBuf,
    /// Directory where ASTRA persists its data (graph, embeddings, vector DB).
    pub data_dir: PathBuf,
    /// File extensions to index.
    pub extensions: Vec<String>,
    /// Embedding provider name ("local" or "openrouter"), overridable via ASTRA_EMBEDDING_PROVIDER.
    pub embedding_provider: String,
}

impl AstraConfig {
    /// Create a new configuration rooted at the given workspace path.
    /// Data is stored in `<workspace>/.folder/ASTRA/`.
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        let workspace_root = workspace_root.as_ref().to_path_buf();
        let data_dir = workspace_root.join(".folder").join("ASTRA");
        let default_provider = if cfg!(feature = "local") {
            "local"
        } else if cfg!(feature = "openrouter") {
            "openrouter"
        } else {
            "none"
        };
        let embedding_provider = std::env::var("ASTRA_EMBEDDING_PROVIDER")
            .unwrap_or_else(|_| default_provider.to_string());

        ensure_gitignore_entry(&workspace_root, ".folder");

        Self {
            workspace_root,
            data_dir,
            extensions: vec![
                "rs".into(),
                "py".into(),
                "js".into(),
                "ts".into(),
                "tsx".into(),
                "jsx".into(),
            ],
            embedding_provider,
        }
    }

    pub fn graph_path(&self) -> PathBuf {
        self.data_dir.join("graph.bin")
    }

    pub fn embeddings_path(&self) -> PathBuf {
        self.data_dir.join("embeddings.bin")
    }

    pub fn vector_db_path(&self) -> PathBuf {
        self.data_dir.join("vector.bin")
    }

    pub fn metadata_path(&self) -> PathBuf {
        self.data_dir.join("metadata.json")
    }
}

fn ensure_gitignore_entry(workspace_root: &Path, entry: &str) {
    let gitignore_path = workspace_root.join(".gitignore");
    let entry_line = entry.to_string();
    if let Ok(content) = std::fs::read_to_string(&gitignore_path) {
        if content.lines().any(|line| line.trim() == entry_line) {
            return;
        }
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&gitignore_path)
    {
        use std::io::Write;
        let _ = writeln!(file, "{}", entry_line);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::AstraConfig;

    #[test]
    fn test_config_paths() {
        let cfg = AstraConfig::new("/tmp/myproject");
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/myproject/.folder/ASTRA"));
        assert_eq!(
            cfg.graph_path(),
            PathBuf::from("/tmp/myproject/.folder/ASTRA/graph.bin")
        );
    }
}
