mod format;
mod transport;
mod types;

pub use types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};

use std::io;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use log::info;
use serde_json::Value;

use crate::search::{ExecutionPath, SearchEngine};

use format::{format_search_results, format_structured_search_results};
use transport::{read_message, write_message};
use types::{
    SearchToolParams, ToolDefinition, FILE_SIZE_CAP, MAX_QUERY_LENGTH, MAX_RESULTS_LIMIT,
    MIN_QUERY_LENGTH,
};

/// MCP server that communicates over stdio.
pub struct McpServer<'a> {
    engine: EngineHandle<'a>,
}

enum EngineHandle<'a> {
    Borrowed(&'a SearchEngine),
    Shared(Arc<RwLock<SearchEngine>>),
}

impl<'a> McpServer<'a> {
    pub fn new(engine: &'a SearchEngine) -> Self {
        Self {
            engine: EngineHandle::Borrowed(engine),
        }
    }

    pub fn new_shared(engine: Arc<RwLock<SearchEngine>>) -> Self {
        Self {
            engine: EngineHandle::Shared(engine),
        }
    }

    /// Run the MCP server, reading JSON-RPC messages from stdin and writing responses to stdout.
    pub fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        info!("MCP server started, listening on stdio");

        loop {
            // Read a message using Content-Length header protocol (LSP-style)
            match read_message(&stdin) {
                Ok(Some(msg)) => {
                    let response = self.handle_message(&msg);
                    if let Some(resp) = response {
                        write_message(&mut stdout, &resp)?;
                    }
                }
                Ok(None) => {
                    // EOF
                    info!("Stdin closed, shutting down MCP server");
                    break;
                }
                Err(e) => {
                    log::error!("Error reading message: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a single JSON-RPC message.
    pub fn handle_message(&self, raw: &str) -> Option<String> {
        let request: JsonRpcRequest = match serde_json::from_str(raw) {
            Ok(req) => req,
            Err(e) => {
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                    }),
                };
                return Some(serde_json::to_string(&resp).unwrap());
            }
        };

        let response = match request.method.as_str() {
            "initialize" => self.handle_initialize(&request),
            "initialized" => {
                // Notification, no response needed
                return None;
            }
            "tools/list" => self.handle_tools_list(&request),
            "tools/call" => self.handle_tools_call(&request),
            "shutdown" | "exit" => {
                return Some(
                    serde_json::to_string(&JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id,
                        result: Some(Value::Null),
                        error: None,
                    })
                    .unwrap(),
                );
            }
            _ => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", request.method),
                }),
            },
        };

        Some(serde_json::to_string(&response).unwrap())
    }

    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let result = serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "tools": {
                    "listChanged": false
                }
            },
            "serverInfo": {
                "name": "ASTra",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(result),
            error: None,
        }
    }

    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tools = vec![
            search_tool_definition(
                "astra_semantic_path_search",
                "Search the codebase semantically. Returns execution paths through the call graph that are most relevant to your query. By default returns the skeleton/discovery map only. Set return_nodes to include full source bodies for selected nodes.",
            ),
            search_tool_definition(
                "astra_semantic_rag_search",
                "Run regular semantic RAG over ASTra's indexed symbol chunks (no path traversal). Use this alongside path-biased search when you want direct chunk retrieval.",
            ),
            search_tool_definition(
                "astra_structured_path_search",
                "Search the codebase semantically and return the execution paths as structured JSON text so downstream agents can reason about scope, terminal nodes, and whether full bodies are needed. By default returns the skeleton/discovery map only. Set return_nodes to include full source bodies for selected nodes.",
            ),
        ];

        let result = serde_json::json!({ "tools": tools });

        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(result),
            error: None,
        }
    }

    fn handle_tools_call(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params = match &request.params {
            Some(p) => p,
            None => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "Missing params".to_string(),
                    }),
                };
            }
        };

        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");

        match tool_name {
            "astra_semantic_path_search"
            | "astra_structured_path_search"
            | "astra_semantic_rag_search" => {
                let (trimmed_query, clamped_max, return_nodes, uncap_file_size) =
                    match validate_search_params(params) {
                        Ok(validated) => validated,
                        Err(error) => {
                            return JsonRpcResponse {
                                jsonrpc: "2.0".to_string(),
                                id: request.id.clone(),
                                result: None,
                                error: Some(error),
                            };
                        }
                    };

                let content = if tool_name == "astra_semantic_rag_search" {
                    match self.engine_search_chunks(&trimmed_query, clamped_max) {
                        Ok(chunks) => {
                            let chunk_paths: Vec<ExecutionPath> = chunks
                                .into_iter()
                                .map(|node| ExecutionPath {
                                    score: node.relevance,
                                    nodes: vec![node],
                                })
                                .collect();
                            format_search_results(&chunk_paths, return_nodes, uncap_file_size)
                        }
                        Err(e) => format!("Error during search: {}", e),
                    }
                } else if tool_name == "astra_structured_path_search" {
                    match self.engine_search(&trimmed_query, clamped_max) {
                        Ok(paths) => {
                            format_structured_search_results(&paths, return_nodes, uncap_file_size)
                        }
                        Err(e) => format!("Error during search: {}", e),
                    }
                } else {
                    match self.engine_search(&trimmed_query, clamped_max) {
                        Ok(paths) => format_search_results(&paths, return_nodes, uncap_file_size),
                        Err(e) => format!("Error during search: {}", e),
                    }
                };

                let result = serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": content
                    }]
                });

                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id.clone(),
                    result: Some(result),
                    error: None,
                }
            }
            _ => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: format!("Unknown tool: {}", tool_name),
                }),
            },
        }
    }

    fn engine_search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<ExecutionPath>> {
        match &self.engine {
            EngineHandle::Borrowed(engine) => engine.search(query, max_results),
            EngineHandle::Shared(engine) => engine
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire engine lock: {}", e))?
                .search(query, max_results),
        }
    }

    fn engine_search_chunks(
        &self,
        query: &str,
        max_results: usize,
    ) -> anyhow::Result<Vec<crate::search::PathNode>> {
        match &self.engine {
            EngineHandle::Borrowed(engine) => engine.search_chunks(query, max_results),
            EngineHandle::Shared(engine) => engine
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire engine lock: {}", e))?
                .search_chunks(query, max_results),
        }
    }
}

fn search_tool_definition(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you're looking for in the codebase"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of execution paths to return (default: 5)",
                    "default": 5
                },
                "return_nodes": {
                    "type": "integer",
                    "description": format!(
                        "Number of path nodes whose full source body to include (default: 0 = skeleton only). \
                        1 = terminal node; 2 = midpoint + terminal; 3 = start + midpoint + terminal; \
                        >3 = the N most-relevant nodes. Bodies are capped to {} chars unless uncap_file_size is true.",
                        FILE_SIZE_CAP
                    ),
                    "default": 0
                },
                "uncap_file_size": {
                    "type": "boolean",
                    "description": format!(
                        "When true, disable the {}-character per-file body cap (default: false).",
                        FILE_SIZE_CAP
                    ),
                    "default": false
                }
            },
            "required": ["query"]
        }),
    }
}

fn validate_search_params(params: &Value) -> Result<(String, usize, usize, bool), JsonRpcError> {
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    let search_params: SearchToolParams =
        serde_json::from_value(arguments).map_err(|e| JsonRpcError {
            code: -32602,
            message: format!("Invalid parameters: {}", e),
        })?;

    let trimmed_query = search_params.query.trim();
    if trimmed_query.len() < MIN_QUERY_LENGTH {
        return Err(JsonRpcError {
            code: -32602,
            message: "Query must not be empty".to_string(),
        });
    }

    if trimmed_query.len() > MAX_QUERY_LENGTH {
        return Err(JsonRpcError {
            code: -32602,
            message: format!(
                "Query too long: {} chars exceeds maximum {}",
                trimmed_query.len(),
                MAX_QUERY_LENGTH
            ),
        });
    }

    Ok((
        trimmed_query.to_string(),
        search_params.max_results.min(MAX_RESULTS_LIMIT),
        search_params.return_nodes,
        search_params.uncap_file_size,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::{McpServer, MAX_QUERY_LENGTH};
    use crate::test_helpers::build_test_engine_from_source;
    use serde_json::Value;

    const SOURCE: &str = r#"
fn handle_request() {
    validate_auth();
}

fn validate_auth() {
    query_database();
}

fn query_database() {
    println!("querying db");
}
"#;

    fn build_test_engine() -> crate::search::SearchEngine {
        build_test_engine_from_source(SOURCE, "server.rs")
    }

    #[test]
    fn test_handle_initialize() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert_eq!(parsed["jsonrpc"], "2.0");
        assert!(parsed["result"]["serverInfo"]["name"] == "ASTra");
    }

    #[test]
    fn test_handle_tools_list() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        let tools = parsed["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);
        let tool_names: Vec<&str> = tools
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert!(tool_names.contains(&"astra_semantic_path_search"));
        assert!(tool_names.contains(&"astra_semantic_rag_search"));
        assert!(tool_names.contains(&"astra_structured_path_search"));
    }

    #[test]
    fn test_handle_tools_call_search() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "astra_semantic_path_search",
                "arguments": {
                    "query": "database query authentication",
                    "max_results": 3
                }
            }
        }"#;

        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert!(parsed["error"].is_null());
        let content = parsed["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            content.contains("Execution Path") || content.contains("No relevant"),
            "should contain search results"
        );
    }

    #[test]
    fn test_handle_tools_call_regular_rag_search() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "astra_semantic_rag_search",
                "arguments": {
                    "query": "database query authentication",
                    "max_results": 2
                }
            }
        }"#;

        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert!(parsed["error"].is_null());
        let content = parsed["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            content.contains("▸") || content.contains("No relevant"),
            "regular rag search should contain chunk results"
        );
    }

    #[test]
    fn test_handle_tools_call_structured_search() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "astra_structured_path_search",
                "arguments": {
                    "query": "request authentication database",
                    "max_results": 1,
                    "return_nodes": 1
                }
            }
        }"#;

        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert!(parsed["error"].is_null());
        let content = parsed["result"]["content"][0]["text"].as_str().unwrap();
        let structured: Value = serde_json::from_str(content).unwrap();
        let paths = structured["paths"].as_array().unwrap();
        assert!(
            !paths.is_empty(),
            "structured search should return at least one path"
        );

        let nodes = paths[0]["nodes"].as_array().unwrap();
        assert!(!nodes.is_empty());
        for node in &nodes[..nodes.len().saturating_sub(1)] {
            assert_eq!(node["is_terminal"], false);
            assert!(node.get("full_body").is_none());
            assert_eq!(node["symbol_kind"], "function");
        }

        let terminal = nodes.last().unwrap();
        assert_eq!(terminal["is_terminal"], true);
        assert!(terminal["full_body"].as_str().is_some());
        assert!(terminal["skeleton"].as_str().is_some());
        assert_eq!(terminal["symbol_kind"], "function");
    }

    #[test]
    fn test_structured_search_default_skeleton_only() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        // Default: return_nodes omitted (defaults to 0) — no full_body anywhere
        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "astra_structured_path_search",
                "arguments": {
                    "query": "request authentication database",
                    "max_results": 1
                }
            }
        }"#;

        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert!(parsed["error"].is_null());
        let content = parsed["result"]["content"][0]["text"].as_str().unwrap();
        let structured: Value = serde_json::from_str(content).unwrap();

        if let Some(paths) = structured["paths"].as_array() {
            for path in paths {
                for node in path["nodes"].as_array().unwrap_or(&vec![]) {
                    assert!(
                        node.get("full_body").is_none(),
                        "default (return_nodes=0) should not include full_body; node: {}",
                        node
                    );
                    assert!(
                        node["skeleton"].is_string(),
                        "all nodes should have a skeleton"
                    );
                }
            }
        }
    }

    #[test]
    fn test_uncap_file_size_flag() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        // return_nodes=1 with uncap_file_size=true — body should be returned without truncation
        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "astra_structured_path_search",
                "arguments": {
                    "query": "request authentication database",
                    "max_results": 1,
                    "return_nodes": 1,
                    "uncap_file_size": true
                }
            }
        }"#;

        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert!(parsed["error"].is_null());
        let content = parsed["result"]["content"][0]["text"].as_str().unwrap();
        let structured: Value = serde_json::from_str(content).unwrap();

        if let Some(first_path) = structured["paths"].as_array().and_then(|p| p.first()) {
            let terminal = first_path["nodes"].as_array().unwrap().last().unwrap();
            assert!(
                terminal["full_body"].is_string(),
                "terminal node should have full_body when return_nodes=1"
            );
            // Bodies in the test fixture are short, so they won't end with "..."
            let body = terminal["full_body"].as_str().unwrap();
            assert!(!body.is_empty());
        }
    }

    #[test]
    fn test_handle_unknown_method() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{"jsonrpc":"2.0","id":4,"method":"unknown/method","params":{}}"#;
        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert!(parsed["error"]["code"].as_i64().unwrap() == -32601);
    }

    #[test]
    fn test_rejects_oversized_query() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let long_query = "a".repeat(MAX_QUERY_LENGTH + 1);
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "astra_semantic_path_search",
                "arguments": {
                    "query": long_query
                }
            }
        })
        .to_string();

        let response = server.handle_message(&msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert_eq!(parsed["error"]["code"].as_i64().unwrap(), -32602);
        assert!(parsed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Query too long"),);
    }

    #[test]
    fn test_rejects_empty_query() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "astra_semantic_path_search",
                "arguments": {
                    "query": "   "
                }
            }
        }"#;

        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert_eq!(parsed["error"]["code"].as_i64().unwrap(), -32602);
        assert_eq!(parsed["error"]["message"], "Query must not be empty");
    }

    #[test]
    fn test_rejects_blank_query() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "astra_semantic_path_search",
                "arguments": {
                    "query": ""
                }
            }
        }"#;

        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert_eq!(parsed["error"]["code"].as_i64().unwrap(), -32602);
        assert_eq!(parsed["error"]["message"], "Query must not be empty");
    }

    #[test]
    fn test_trims_query_before_search() {
        let engine = build_test_engine();
        let server = McpServer::new(&engine);

        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "astra_semantic_path_search",
                "arguments": {
                    "query": "   database query authentication   ",
                    "max_results": 1
                }
            }
        }"#;

        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();

        assert!(
            parsed["error"].is_null(),
            "trimmed query should be accepted"
        );
    }

    #[test]
    fn test_new_shared_server_handles_search() {
        let engine = Arc::new(RwLock::new(build_test_engine()));
        let server = McpServer::new_shared(engine);
        let msg = r#"{
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "astra_semantic_path_search",
                "arguments": {
                    "query": "database query authentication",
                    "max_results": 1
                }
            }
        }"#;
        let response = server.handle_message(msg).unwrap();
        let parsed: Value = serde_json::from_str(&response).unwrap();
        assert!(parsed["error"].is_null());
    }
}
