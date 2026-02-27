use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 request.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

/// Tool definition for the MCP protocol.
#[derive(Debug, Serialize)]
pub(crate) struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Parameters for the astra_semantic_path_search tool.
#[derive(Debug, Deserialize)]
pub(crate) struct SearchToolParams {
    pub query: String,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    /// Number of path nodes whose full source body to include in the response.
    ///
    /// - 0 (default): return only the skeleton/discovery map for all nodes.
    /// - 1: include the full body of the terminal (target) node.
    /// - 2: include the full body of the midpoint and terminal nodes.
    /// - 3: include the full body of the start, midpoint, and terminal nodes.
    /// - >3: include the full body of the `return_nodes` most-relevant nodes.
    #[serde(default)]
    pub return_nodes: usize,
    /// When `true`, disable the 32,000-character per-file body cap.
    #[serde(default)]
    pub uncap_file_size: bool,
}

pub(crate) fn default_max_results() -> usize {
    5
}

/// Hard upper bound for max_results to prevent resource exhaustion.
pub(crate) const MAX_RESULTS_LIMIT: usize = 50;

/// Maximum allowed query length in characters to prevent CPU exhaustion in the embedder.
pub(crate) const MAX_QUERY_LENGTH: usize = 10_000;
/// Minimum query length after trimming surrounding whitespace.
pub(crate) const MIN_QUERY_LENGTH: usize = 1;

/// Default per-file body character cap when `uncap_file_size` is false.
pub(crate) const FILE_SIZE_CAP: usize = 32_000;
