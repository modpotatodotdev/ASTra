use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use tempfile::TempDir;

use astra::config::AstraConfig;
use astra::embeddings::{build_embedder, DEFAULT_LOCAL_DIM};
use astra::indexer;
use astra::mcp::McpServer;
use astra::search::SearchEngine;
use astra::storage;

/// Helper: create a workspace with sample Rust files and return the temp dir + config.
fn setup_rust_workspace() -> (TempDir, AstraConfig) {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();

    // Main entry point that routes requests
    fs::write(
        src.join("main.rs"),
        r#"
fn main() {
    let app = create_router();
    app.serve();
}

fn create_router() {
    handle_request();
}

fn handle_request() {
    auth_middleware();
}
"#,
    )
    .unwrap();

    // Auth middleware
    fs::write(
        src.join("auth.rs"),
        r#"
fn auth_middleware() {
    let token = extract_token();
    validate_token(token);
}

fn extract_token() -> String {
    "bearer abc123".to_string()
}

fn validate_token(token: &str) -> bool {
    token.starts_with("bearer")
}
"#,
    )
    .unwrap();

    // Database layer
    fs::write(
        src.join("db.rs"),
        r#"
fn db_insert(data: &str) {
    execute_query(data);
}

fn execute_query(query: &str) {
    println!("executing: {}", query);
}

fn db_select(table: &str) -> Vec<String> {
    let result = execute_query(table);
    vec![]
}
"#,
    )
    .unwrap();

    let config = AstraConfig::new(tmp.path());
    (tmp, config)
}

/// Helper: create a workspace with Python files.
fn setup_python_workspace() -> (TempDir, AstraConfig) {
    let tmp = TempDir::new().unwrap();

    fs::write(
        tmp.path().join("app.py"),
        r#"
class UserService:
    def create_user(self, name, email):
        user = self.validate_input(name, email)
        return self.save_to_db(user)

    def validate_input(self, name, email):
        if not name:
            raise ValueError("name required")
        return {"name": name, "email": email}

    def save_to_db(self, user):
        return user

def handle_signup(name, email):
    service = UserService()
    return service.create_user(name, email)
"#,
    )
    .unwrap();

    let config = AstraConfig::new(tmp.path());
    (tmp, config)
}

/// Helper: create a workspace with JavaScript files.
fn setup_js_workspace() -> (TempDir, AstraConfig) {
    let tmp = TempDir::new().unwrap();

    fs::write(
        tmp.path().join("api.js"),
        r#"
function fetchUser(userId) {
    return queryDatabase(userId);
}

function queryDatabase(id) {
    return { id: id, name: "test" };
}

function renderProfile(userId) {
    const user = fetchUser(userId);
    return formatOutput(user);
}

function formatOutput(data) {
    return JSON.stringify(data);
}
"#,
    )
    .unwrap();

    let config = AstraConfig::new(tmp.path());
    (tmp, config)
}

// ========== E2E Tests ==========

#[test]
fn e2e_cold_start_indexing_rust() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();

    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();

    assert_eq!(result.files_indexed, 3, "should index 3 Rust files");
    assert!(
        result.symbols_indexed >= 8,
        "should index at least 8 symbols, got {}",
        result.symbols_indexed
    );
    assert!(result.graph.node_count() >= 8);
    assert!(result.graph.edge_count() > 0, "should have call edges");

    // Verify persistence
    assert!(config.graph_path().exists());
    assert!(config.vector_db_path().exists());
    assert!(config.metadata_path().exists());
}

#[test]
fn e2e_cold_start_indexing_python() {
    let (_tmp, config) = setup_python_workspace();
    let embedder = build_embedder(config.embedding_provider.as_str()).unwrap();

    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();

    assert_eq!(result.files_indexed, 1);
    // Should find: UserService class, create_user, validate_input, save_to_db, handle_signup
    assert!(
        result.symbols_indexed >= 4,
        "should index Python symbols, got {}",
        result.symbols_indexed
    );
}

#[test]
fn e2e_cold_start_indexing_javascript() {
    let (_tmp, config) = setup_js_workspace();
    let embedder = build_embedder(config.embedding_provider.as_str()).unwrap();

    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();

    assert_eq!(result.files_indexed, 1);
    assert!(
        result.symbols_indexed >= 4,
        "should index JS functions, got {}",
        result.symbols_indexed
    );
}

#[test]
fn e2e_semantic_search_finds_relevant_path() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder(config.embedding_provider.as_str()).unwrap();

    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);

    // Search for database-related functionality
    let results = engine.search("database insert query", 5).unwrap();

    assert!(
        !results.is_empty(),
        "should find results for database query"
    );

    // Verify the results contain relevant symbols
    let all_names: Vec<&str> = results
        .iter()
        .flat_map(|p| p.nodes.iter().map(|n| n.name.as_str()))
        .collect();

    assert!(
        all_names
            .iter()
            .any(|n| n.contains("db") || n.contains("query") || n.contains("execute")),
        "should find database-related symbols, got: {:?}",
        all_names
    );
}

#[test]
fn e2e_semantic_search_auth_flow() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder(config.embedding_provider.as_str()).unwrap();

    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);

    let results = engine.search("authentication token validation", 5).unwrap();
    let all_names: Vec<&str> = results
        .iter()
        .flat_map(|p| p.nodes.iter().map(|n| n.name.as_str()))
        .collect();

    assert!(
        all_names
            .iter()
            .any(|n| n.contains("auth") || n.contains("token") || n.contains("validate")),
        "should find auth-related symbols, got: {:?}",
        all_names
    );
}

#[test]
fn e2e_persistence_and_reload() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();

    // Index
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let original_node_count = result.graph.node_count();

    // Reload from disk
    let mut loaded_graph = storage::load_graph(&config).unwrap();
    loaded_graph.rebuild_after_deserialize();
    let loaded_vectors = storage::load_vector_store(&config).unwrap();

    assert_eq!(loaded_graph.node_count(), original_node_count);
    assert_eq!(loaded_vectors.len(), original_node_count);

    // Verify search still works after reload
    let engine = SearchEngine::new(loaded_graph, loaded_vectors, embedder);
    let results = engine.search("router request handling", 3).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn e2e_incremental_update() {
    let (tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();

    let mut result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let original_count = result.graph.node_count();

    // Add a new file
    fs::write(
        tmp.path().join("src").join("utils.rs"),
        r#"
fn format_date(timestamp: u64) -> String {
    format!("date: {}", timestamp)
}

fn parse_json(input: &str) -> String {
    input.to_string()
}
"#,
    )
    .unwrap();

    // Incremental update
    let updated = indexer::update_files(
        &config,
        &mut result.graph,
        &mut result.vector_store,
        embedder.as_ref(),
        &["src/utils.rs".to_string()],
    )
    .unwrap();

    assert!(updated >= 2, "should add at least 2 new symbols");
    assert!(
        result.graph.node_count() > original_count,
        "graph should grow after update"
    );
}

#[test]
fn e2e_file_deletion_update() {
    let (tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();

    let mut result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let original_count = result.graph.node_count();

    // Delete a file
    fs::remove_file(tmp.path().join("src").join("db.rs")).unwrap();

    // Update for deleted file
    let _updated = indexer::update_files(
        &config,
        &mut result.graph,
        &mut result.vector_store,
        embedder.as_ref(),
        &["src/db.rs".to_string()],
    )
    .unwrap();

    assert!(
        result.graph.node_count() < original_count,
        "graph should shrink after file deletion"
    );
}

#[test]
fn e2e_mcp_initialize() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);
    let server = McpServer::new(&engine);

    // Test initialize
    let msg = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{}}}"#;
    let response = server.handle_message(msg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    assert_eq!(parsed["result"]["serverInfo"]["name"], "ASTra");
    assert!(parsed["result"]["capabilities"]["tools"].is_object());
    assert_eq!(parsed["result"]["protocolVersion"], "2025-11-25");
}

#[test]
fn e2e_mcp_tools_list() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);
    let server = McpServer::new(&engine);

    let msg = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
    let response = server.handle_message(msg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    let tools = parsed["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 3);
    let tool_names: Vec<&str> = tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert!(tool_names.contains(&"astra_semantic_path_search"));
    assert!(tool_names.contains(&"astra_semantic_rag_search"));
    assert!(tool_names.contains(&"astra_structured_path_search"));
    for tool in tools {
        assert!(tool["inputSchema"]["properties"]["query"].is_object());
    }
}

#[test]
fn e2e_mcp_semantic_search_tool() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);
    let server = McpServer::new(&engine);

    let msg = r#"{
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "astra_semantic_path_search",
            "arguments": {
                "query": "authentication middleware token validation",
                "max_results": 3
            }
        }
    }"#;

    let response = server.handle_message(msg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    assert!(parsed["error"].is_null(), "should not have error");
    let content = parsed["result"]["content"][0]["text"].as_str().unwrap();

    // Should return execution paths
    assert!(
        content.contains("Execution Path") || content.contains("No relevant"),
        "should contain search results, got: {}",
        &content[..content.len().min(200)]
    );
}

#[test]
fn e2e_mcp_regular_rag_search_tool() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);
    let server = McpServer::new(&engine);

    let msg = r#"{
        "jsonrpc": "2.0",
        "id": 35,
        "method": "tools/call",
        "params": {
            "name": "astra_semantic_rag_search",
            "arguments": {
                "query": "authentication middleware token validation",
                "max_results": 3
            }
        }
    }"#;

    let response = server.handle_message(msg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    assert!(parsed["error"].is_null(), "should not have error");
    let content = parsed["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        content.contains("▸") || content.contains("No relevant"),
        "should contain search results, got: {}",
        &content[..content.len().min(200)]
    );
}

#[test]
fn e2e_mcp_structured_search_tool() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);
    let server = McpServer::new(&engine);

    // Default request: return_nodes omitted → skeleton-only, no full_body
    let msg_default = r#"{
        "jsonrpc": "2.0",
        "id": 33,
        "method": "tools/call",
        "params": {
            "name": "astra_structured_path_search",
            "arguments": {
                "query": "authentication middleware token validation",
                "max_results": 2
            }
        }
    }"#;

    let response = server.handle_message(msg_default).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    assert!(parsed["error"].is_null(), "should not have error");
    let content = parsed["result"]["content"][0]["text"].as_str().unwrap();
    let structured: serde_json::Value = serde_json::from_str(content).unwrap();

    assert!(structured["paths"].is_array());
    if let Some(first_path) = structured["paths"]
        .as_array()
        .and_then(|paths| paths.first())
    {
        let nodes = first_path["nodes"].as_array().unwrap();
        assert!(!nodes.is_empty());
        // Default: all nodes should have skeleton but no full_body
        assert!(nodes.iter().all(|node| node["skeleton"].is_string()));
        assert!(
            nodes.iter().all(|node| node.get("full_body").is_none()),
            "default (return_nodes=0) should not include full_body in any node"
        );
    }

    // return_nodes=1 → terminal node should have full_body
    let msg_with_nodes = r#"{
        "jsonrpc": "2.0",
        "id": 34,
        "method": "tools/call",
        "params": {
            "name": "astra_structured_path_search",
            "arguments": {
                "query": "authentication middleware token validation",
                "max_results": 2,
                "return_nodes": 1
            }
        }
    }"#;

    let response2 = server.handle_message(msg_with_nodes).unwrap();
    let parsed2: serde_json::Value = serde_json::from_str(&response2).unwrap();

    assert!(parsed2["error"].is_null(), "should not have error");
    let content2 = parsed2["result"]["content"][0]["text"].as_str().unwrap();
    let structured2: serde_json::Value = serde_json::from_str(content2).unwrap();

    if let Some(first_path) = structured2["paths"]
        .as_array()
        .and_then(|paths| paths.first())
    {
        let nodes = first_path["nodes"].as_array().unwrap();
        assert!(!nodes.is_empty());

        let terminal = nodes.last().unwrap();
        assert_eq!(terminal["is_terminal"], true);
        assert!(terminal["full_body"].is_string());

        for node in &nodes[..nodes.len().saturating_sub(1)] {
            assert_eq!(node["is_terminal"], false);
            assert!(node.get("full_body").is_none());
        }
    }
}

#[test]
fn e2e_mcp_stdio_binary_smoke() {
    let (tmp, _config) = setup_rust_workspace();
    let binary = env!("CARGO_BIN_EXE_astra");

    let mut child = Command::new(binary)
        .arg(tmp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to start ASTra binary");

    let mut stdin = child.stdin.take().expect("failed to capture stdin");
    let stdout = child.stdout.take().expect("failed to capture stdout");
    let mut stdout_reader = BufReader::new(stdout);

    let mut send = |request: &str| -> serde_json::Value {
        writeln!(stdin, "{request}").expect("failed to write request");
        stdin.flush().expect("failed to flush request");

        let mut line = String::new();
        stdout_reader
            .read_line(&mut line)
            .expect("failed to read response line");
        assert!(!line.trim().is_empty(), "response must not be empty");

        serde_json::from_str(line.trim()).expect("response must be valid JSON")
    };

    let init = send(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{}}}"#,
    );
    assert_eq!(init["result"]["serverInfo"]["name"], "ASTra");
    assert_eq!(init["result"]["protocolVersion"], "2025-11-25");

    let tools = send(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#);
    let tool_names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert!(tool_names.contains(&"astra_semantic_path_search"));
    assert!(tool_names.contains(&"astra_semantic_rag_search"));
    assert!(tool_names.contains(&"astra_structured_path_search"));

    let result = send(
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"astra_semantic_path_search","arguments":{"query":"authentication middleware token validation","max_results":2}}}"#,
    );
    assert!(
        result["error"].is_null(),
        "search should not return JSON-RPC error"
    );
    let content = result["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        content.contains("Execution Path") || content.contains("No relevant"),
        "expected path search content, got: {}",
        &content[..content.len().min(200)]
    );

    let shutdown = send(r#"{"jsonrpc":"2.0","id":4,"method":"shutdown","params":{}}"#);
    assert!(shutdown["error"].is_null());

    drop(stdin);
    let status = child.wait().expect("failed to wait for ASTra process");
    assert!(
        status.success(),
        "ASTra process exited with status: {status}"
    );
}

#[test]
fn e2e_mcp_search_with_execution_path() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);
    let server = McpServer::new(&engine);

    let msg = r#"{
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "astra_semantic_path_search",
            "arguments": {
                "query": "database execute query insert",
                "max_results": 5
            }
        }
    }"#;

    let response = server.handle_message(msg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    let content = parsed["result"]["content"][0]["text"].as_str().unwrap();
    // The path notation should show arrows
    if content.contains("Execution Path") {
        assert!(
            content.contains("→") || content.contains("->"),
            "execution paths should show arrow notation"
        );
    }
}

#[test]
fn e2e_cross_language_indexing() {
    let tmp = TempDir::new().unwrap();

    // Rust file
    fs::write(
        tmp.path().join("lib.rs"),
        "fn rust_func() { println!(\"hello\"); }\n",
    )
    .unwrap();

    // Python file
    fs::write(
        tmp.path().join("app.py"),
        "def python_func():\n    print('hello')\n",
    )
    .unwrap();

    // JavaScript file
    fs::write(
        tmp.path().join("index.js"),
        "function jsFunc() { console.log('hello'); }\n",
    )
    .unwrap();

    let config = AstraConfig::new(tmp.path());
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();

    assert_eq!(result.files_indexed, 3);
    assert!(result.symbols_indexed >= 3);

    // Each language's function should be searchable
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);

    let rust_results = engine.search("rust function", 3).unwrap();
    let python_results = engine.search("python function", 3).unwrap();
    let js_results = engine.search("javascript function console", 3).unwrap();

    assert!(!rust_results.is_empty());
    assert!(!python_results.is_empty());
    assert!(!js_results.is_empty());
}

#[test]
fn e2e_empty_workspace() {
    let tmp = TempDir::new().unwrap();
    let config = AstraConfig::new(tmp.path());
    let embedder = build_embedder("local").unwrap();

    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();

    assert_eq!(result.files_indexed, 0);
    assert_eq!(result.symbols_indexed, 0);

    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);
    let results = engine.search("anything", 5).unwrap();
    assert!(results.is_empty());
}

#[test]
fn e2e_storage_data_directory() {
    let tmp = TempDir::new().unwrap();
    let config = AstraConfig::new(tmp.path());

    // Data dir should be at .folder/ASTra/
    assert!(config.data_dir.to_string_lossy().contains(".folder/ASTra"));

    let embedder = build_embedder("local").unwrap();
    indexer::index_workspace(&config, embedder.as_ref()).unwrap();

    // Verify the .folder/ASTra directory was created
    assert!(config.data_dir.exists());
    assert!(config.data_dir.join("graph.bin").exists());
    assert!(config.data_dir.join("vector.bin").exists());
    assert!(config.data_dir.join("metadata.json").exists());
}

#[test]
fn e2e_clear_and_reindex() {
    let (_tmp, config) = setup_rust_workspace();
    let embedder = build_embedder("local").unwrap();

    // First index
    indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    assert!(storage::has_persisted_data(&config));

    // Clear data
    storage::clear_data(&config).unwrap();
    assert!(!storage::has_persisted_data(&config));

    // Re-index
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    assert!(result.files_indexed > 0);
    assert!(storage::has_persisted_data(&config));
}

// ── Tests for the three fixes introduced in the previous PR ──────────────────

/// Fix 1 – symbol deduplication: when multiple execution paths share the same
/// intermediate node, that node must appear at most once in the assembled
/// context string.
///
/// We replicate the same deduplication logic used by `run_astra` in the
/// benchmark binary: iterate over paths, skip any node whose `symbol_id` has
/// already been seen.
#[test]
fn e2e_astra_deduplicates_symbols_in_context() {
    // Build a workspace where func_a and func_b *both* call shared_impl,
    // so two separate top-level entry-point searches are likely to produce
    // paths that both traverse shared_impl.
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("shared.rs"),
        r#"
fn func_a() {
    shared_impl();
}

fn func_b() {
    shared_impl();
}

fn shared_impl() {
    println!("shared database write");
}
"#,
    )
    .unwrap();

    let config = AstraConfig::new(tmp.path());
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);

    // Search with a generous top_k so we get multiple overlapping paths
    let paths = engine.search("database write shared", 10).unwrap();
    assert!(!paths.is_empty(), "should find at least one path");

    // Replicate run_astra deduplication
    let mut seen_symbols = std::collections::HashSet::new();
    let mut context = String::new();
    for path in paths {
        for node in path.nodes {
            if !seen_symbols.insert(node.symbol_id.clone()) {
                // Skip duplicate – do NOT append to context
                continue;
            }
            context.push_str(&node.body);
        }
    }

    // Every symbol_id must appear at most once in the context
    let symbol_count = seen_symbols.len();
    // Count how many times "shared_impl" body text appears (proxy for duplicates)
    let occurrences = context.matches("shared database write").count();
    assert_eq!(
        occurrences, 1,
        "shared_impl body should appear exactly once in context (got {}), symbol count: {}",
        occurrences, symbol_count
    );
}

/// Fix 2 – increased A* depth: the search engine must be able to traverse a
/// call chain that is longer than the previous hard limit of 3 hops (now 10).
///
/// We build a 6-deep chain: entry → l1 → l2 → l3 → l4 → deep_target and
/// assert that `deep_target` appears in search results.
#[test]
fn e2e_astar_finds_deep_call_chain() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("chain.rs"),
        r#"
fn entry_point() {
    level_one();
}

fn level_one() {
    level_two();
}

fn level_two() {
    level_three();
}

fn level_three() {
    level_four();
}

fn level_four() {
    deep_target();
}

fn deep_target() {
    println!("deep database migration bug");
}
"#,
    )
    .unwrap();

    let config = AstraConfig::new(tmp.path());
    let embedder = build_embedder("local").unwrap();
    let result = indexer::index_workspace(&config, embedder.as_ref()).unwrap();
    let engine = SearchEngine::new(result.graph, result.vector_store, embedder);

    let paths = engine.search("deep database migration bug", 5).unwrap();
    assert!(!paths.is_empty(), "should find at least one path");

    let all_names: Vec<&str> = paths
        .iter()
        .flat_map(|p| p.nodes.iter().map(|n| n.name.as_str()))
        .collect();

    assert!(
        all_names.contains(&"deep_target"),
        "search should reach deep_target (6 hops from entry_point), found: {:?}",
        all_names
    );
}

/// Fix 3 – SemanticEmbedder produces 768-dimensional embeddings (BGE-base-en-v1.5)
/// and is the only embedder in the codebase.
#[test]
fn e2e_semantic_embedder_dim() {
    let embedder = build_embedder("local").expect("model should load");
    assert_eq!(
        embedder.dim(),
        DEFAULT_LOCAL_DIM,
        "SemanticEmbedder must produce 768-dim vectors"
    );
    let vec = embedder
        .embed("fn compute_total() -> f64 { 42.0 }")
        .unwrap();
    assert_eq!(vec.len(), DEFAULT_LOCAL_DIM);
}
