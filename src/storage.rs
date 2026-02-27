use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use bincode::Options;
use serde::{Deserialize, Serialize};

use crate::config::AstraConfig;
use crate::graph::CallGraph;
use crate::vector_store::VectorStore;

/// Metadata about the last indexing run, used for incremental updates.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexMetadata {
    /// Map from file path (relative to workspace root) to last-modified timestamp.
    pub file_timestamps: HashMap<String, u64>,
}

/// Maximum allowed size for deserialized data structures (512 MiB).
/// Prevents memory exhaustion from crafted binary files.
const MAX_DESERIALIZE_SIZE: u64 = 512 * 1024 * 1024;

/// Create bincode options with a size limit to prevent memory exhaustion.
fn bincode_options() -> impl Options {
    bincode::options().with_limit(MAX_DESERIALIZE_SIZE)
}

/// Persist the call graph to disk, zstd-compressed at level 1.
pub fn save_graph(config: &AstraConfig, graph: &CallGraph) -> Result<()> {
    fs::create_dir_all(&config.data_dir)?;
    let serialized = bincode_options()
        .serialize(graph)
        .context("failed to serialize graph")?;
    // Level 1 is the zstd "fast" preset: still gives ~3-5x compression on
    // text-heavy graph data with decompression overhead well under 1 ms.
    let compressed = zstd::encode_all(&serialized[..], 1).context("failed to compress graph")?;
    fs::write(config.graph_path(), compressed).context("failed to write graph")?;
    Ok(())
}

/// Load the call graph from disk.
/// Transparently handles both zstd-compressed (current) and raw bincode (legacy) files.
/// signals a legacy file rather than silent corruption.
pub fn load_graph(config: &AstraConfig) -> Result<CallGraph> {
    let raw = fs::read(config.graph_path()).context("failed to read graph")?;
    let bytes = match zstd::decode_all(&raw[..]) {
        Ok(decompressed) => decompressed,
        Err(_) => raw,
    };
    bincode_options()
        .deserialize(&bytes)
        .context("failed to deserialize graph")
}

/// Persist the vector store to disk.
pub fn save_vector_store(config: &AstraConfig, store: &VectorStore) -> Result<()> {
    fs::create_dir_all(&config.data_dir)?;
    let encoded = bincode_options()
        .serialize(store)
        .context("failed to serialize vector store")?;
    fs::write(config.vector_db_path(), encoded).context("failed to write vector store")?;
    Ok(())
}

/// Load the vector store from disk.
pub fn load_vector_store(config: &AstraConfig) -> Result<VectorStore> {
    let bytes = fs::read(config.vector_db_path()).context("failed to read vector store")?;
    bincode_options()
        .deserialize(&bytes)
        .context("failed to deserialize vector store")
}

/// Persist index metadata (file timestamps) to disk.
pub fn save_metadata(config: &AstraConfig, metadata: &IndexMetadata) -> Result<()> {
    fs::create_dir_all(&config.data_dir)?;
    let json = serde_json::to_string_pretty(metadata)?;
    fs::write(config.metadata_path(), json)?;
    Ok(())
}

/// Load index metadata from disk.
pub fn load_metadata(config: &AstraConfig) -> Result<IndexMetadata> {
    let path = config.metadata_path();
    if !path.exists() {
        return Ok(IndexMetadata::default());
    }
    let json = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}

/// Check if ASTra data exists on disk for the given workspace.
pub fn has_persisted_data(config: &AstraConfig) -> bool {
    config.graph_path().exists() && config.vector_db_path().exists()
}

/// Delete all persisted data.
pub fn clear_data(config: &AstraConfig) -> Result<()> {
    if config.data_dir.exists() {
        fs::remove_dir_all(&config.data_dir)?;
    }
    Ok(())
}

/// Ensure the data directory exists.
pub fn ensure_data_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{load_metadata, save_metadata, IndexMetadata};
    use crate::config::AstraConfig;
    use tempfile::TempDir;

    #[test]
    fn test_metadata_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let cfg = AstraConfig::new(tmp.path());

        let mut meta = IndexMetadata::default();
        meta.file_timestamps.insert("src/main.rs".into(), 12345);

        save_metadata(&cfg, &meta).unwrap();
        let loaded = load_metadata(&cfg).unwrap();
        assert_eq!(loaded.file_timestamps.get("src/main.rs"), Some(&12345));
    }
}
