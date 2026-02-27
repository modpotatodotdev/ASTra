use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use astra::config::AstraConfig;
use astra::indexer;
use astra::search::SearchEngine;
use llmg_core::provider::{Provider, ProviderRegistry, RoutingProvider};
use llmg_core::types::{ChatCompletionRequest, FunctionDefinition, Message, Tool, ToolChoice};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{Mutex, Semaphore};

// ─── CLI args ────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Args {
    workspace: PathBuf,
    swe_bench_jsonl: PathBuf,
    output: PathBuf,
    metrics_output: PathBuf,
    max_cases: usize,
    top_k: usize,
    llmg_model: String,
    llmg_max_tokens: u32,
    max_tool_calls: usize,
    concurrency: usize,
}

// ─── SWE-bench case ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SweBenchCase {
    id: String,
    repo: String,
    query: String,
}

// ─── Per-instance output ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct PredictionRecord {
    instance_id: String,
    model_patch: String,
    total_tool_calls_used: usize,
    tool_call_breakdown: HashMap<String, usize>,
    prompt_tokens: u64,
    completion_tokens: u64,
}

// ─── Metrics summary ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct MetricsSummary {
    total_cases: usize,
    completed: usize,
    failed_or_incomplete: usize,
    avg_tool_calls: f64,
    avg_prompt_tokens: f64,
    avg_completion_tokens: f64,
    tool_call_totals: HashMap<String, usize>,
}

// ─── LLM agent ───────────────────────────────────────────────────────────────

#[derive(Debug)]
struct MultiTurnAgent {
    client: RoutingProvider,
    model: String,
    max_tokens: u32,
}

impl MultiTurnAgent {
    async fn from_env(model: String, max_tokens: u32) -> Result<Self> {
        let mut registry = ProviderRegistry::new();
        let provider_prefix = model.split('/').next().unwrap_or("");
        match provider_prefix {
            "z_ai" => {
                if let Ok(client) = llmg_providers::z_ai::ZaiClient::from_env() {
                    registry.register(Arc::new(client));
                }
            }
            "z_ai_coding" => {
                if let Ok(client) = llmg_providers::z_ai::ZaiClient::coding_from_env() {
                    registry.register(Arc::new(client));
                }
            }
            _ => {
                if let Ok(client) = llmg_providers::openai::OpenAiClient::from_env() {
                    registry.register(Arc::new(client));
                }
                let client = llmg_providers::ollama::OllamaClient::from_env();
                registry.register(Arc::new(client));
            }
        }
        if registry.list().is_empty() {
            return Err(anyhow!(
                "No LLMG providers are configured via environment variables for model prefix '{}'",
                provider_prefix
            ));
        }
        Ok(Self {
            client: RoutingProvider::new(registry),
            model,
            max_tokens,
        })
    }

    /// Send a single chat completion request with exponential backoff retries.
    /// Returns (response_message, prompt_tokens, completion_tokens).
    async fn chat(&self, messages: Vec<Message>, tools: &[Tool]) -> Result<(Message, u64, u64)> {
        for attempt in 0u32..5 {
            let request = ChatCompletionRequest {
                model: self.model.clone(),
                messages: messages.clone(),
                temperature: Some(0.0),
                max_tokens: Some(self.max_tokens),
                stream: Some(false),
                top_p: None,
                frequency_penalty: None,
                presence_penalty: None,
                stop: None,
                user: Some("astra-multi-turn-tool-benchmark".to_string()),
                tools: if tools.is_empty() {
                    None
                } else {
                    Some(tools.to_vec())
                },
                tool_choice: if tools.is_empty() {
                    None
                } else {
                    Some(ToolChoice::String("auto".to_string()))
                },
            };

            match self.client.chat_completion(request).await {
                Ok(response) => {
                    let (pt, ct) = if let Some(usage) = &response.usage {
                        (usage.prompt_tokens as u64, usage.completion_tokens as u64)
                    } else {
                        (0u64, 0u64)
                    };
                    let msg = response
                        .choices
                        .into_iter()
                        .next()
                        .map(|c| c.message)
                        .ok_or_else(|| anyhow!("LLM returned no choices"))?;
                    return Ok((msg, pt, ct));
                }
                Err(e) => {
                    let wait = 2_u64.pow(attempt);
                    println!(
                        "  LLM attempt {} failed: {:?}. Retrying in {}s…",
                        attempt + 1,
                        e,
                        wait
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs(wait)).await;
                }
            }
        }
        Err(anyhow!("All LLM retry attempts exhausted"))
    }
}

// ─── Tool definitions (JSON schema) ──────────────────────────────────────────

fn build_tools() -> Vec<Tool> {
    vec![
        Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "search_astra".to_string(),
                description: Some(
                    "Search the codebase using ASTra semantic search. \
                     Returns the most relevant code symbols and snippets."
                        .to_string(),
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query describing the code you are looking for"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "run_grep".to_string(),
                description: Some(
                    "Run grep to search for a pattern in the repository. \
                     Returns matching lines."
                        .to_string(),
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The grep pattern to search for"
                        },
                        "dir": {
                            "type": "string",
                            "description": "Subdirectory within the repository to search in (default: . = root)"
                        }
                    },
                    "required": ["pattern"]
                }),
            },
        },
        Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "run_ripgrep".to_string(),
                description: Some(
                    "Run ripgrep (rg) to search for a pattern in the repository. \
                     Faster and more expressive than grep."
                        .to_string(),
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The ripgrep pattern to search for"
                        },
                        "dir": {
                            "type": "string",
                            "description": "Subdirectory within the repository to search in (default: . = root)"
                        }
                    },
                    "required": ["pattern"]
                }),
            },
        },
        Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "submit_patch".to_string(),
                description: Some(
                    "Submit the final unified diff patch to fix the issue. \
                     This ends the session. Call this only when you are ready to submit your fix."
                        .to_string(),
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "diff": {
                            "type": "string",
                            "description": "The complete unified diff (patch) to fix the issue, \
                                            with a/ and b/ prefixes, starting with --- and +++."
                        }
                    },
                    "required": ["diff"]
                }),
            },
        },
    ]
}

// ─── System prompt ────────────────────────────────────────────────────────────

fn build_system_prompt(max_tool_calls: usize) -> String {
    format!(
        "You are an expert software engineer resolving GitHub issues.\n\
         You have access to a set of tools to search the codebase. \
         Use as few tool calls as possible — you have a strict limit of {} tool call(s) total.\n\
         \n\
         Available tools:\n\
         - search_astra(query, top_k): Semantic search over the codebase using ASTra.\n\
         - run_grep(pattern, dir): Search with grep.\n\
         - run_ripgrep(pattern, dir): Search with ripgrep (rg).\n\
         - submit_patch(diff): Submit your unified diff patch to fix the issue. \
           This MUST be your last action.\n\
         \n\
         Instructions:\n\
         1. Read the issue carefully.\n\
         2. Use tools to find the relevant code (minimize tool calls).\n\
         3. When you know the fix, call submit_patch with the complete unified diff.\n\
         4. If you cannot determine the fix, still call submit_patch with an empty string.\n\
         \n\
         The diff must be a standard unified diff applied from the repository root \
         (e.g., --- a/src/main.py, +++ b/src/main.py).",
        max_tool_calls
    )
}

// ─── Tool execution ───────────────────────────────────────────────────────────

/// Execute a tool call and return its text output (truncated at 8 KB).
fn execute_tool(
    name: &str,
    args: &Value,
    workspace_dir: &Path,
    engine: &SearchEngine,
    top_k: usize,
) -> String {
    match name {
        "search_astra" => {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let k = args
                .get("top_k")
                .and_then(Value::as_u64)
                .unwrap_or(top_k as u64) as usize;
            if query.is_empty() {
                return "Error: query is required".to_string();
            }
            run_astra_search(engine, &query, k)
        }
        "run_grep" => {
            let pattern = args
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let dir = args.get("dir").and_then(Value::as_str).unwrap_or(".");
            if pattern.is_empty() {
                return "Error: pattern is required".to_string();
            }
            let safe_dir = match resolve_tool_dir(dir, workspace_dir) {
                Ok(safe_dir) => safe_dir,
                Err(msg) => return msg,
            };
            let safe_dir_str = safe_dir.to_string_lossy().to_string();
            run_command_in_workspace(
                "grep",
                &[
                    "-r",
                    "-n",
                    "--include=*.py",
                    "--include=*.rs",
                    "--include=*.js",
                    "--include=*.ts",
                    "--include=*.java",
                    "--include=*.go",
                    "--include=*.c",
                    "--include=*.cpp",
                    "--include=*.h",
                    "-l",
                    &pattern,
                    &safe_dir_str,
                ],
                workspace_dir,
            )
        }
        "run_ripgrep" => {
            let pattern = args
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let dir = args.get("dir").and_then(Value::as_str).unwrap_or(".");
            if pattern.is_empty() {
                return "Error: pattern is required".to_string();
            }
            let safe_dir = match resolve_tool_dir(dir, workspace_dir) {
                Ok(safe_dir) => safe_dir,
                Err(msg) => return msg,
            };
            let safe_dir_str = safe_dir.to_string_lossy().to_string();
            run_command_in_workspace("rg", &["-n", &pattern, &safe_dir_str], workspace_dir)
        }
        "submit_patch" => {
            // Handled specially in the turn loop; should not reach here.
            "Patch submitted.".to_string()
        }
        _ => format!("Unknown tool: {}", name),
    }
}

/// Resolve a tool-provided directory path against the workspace and reject
/// attempts to escape it.
fn resolve_tool_dir(dir: &str, workspace_dir: &Path) -> std::result::Result<PathBuf, String> {
    let requested = Path::new(dir);
    if requested.is_absolute() {
        return Err("Error: absolute paths are not allowed".to_string());
    }

    let mut normalized = PathBuf::new();
    for component in requested.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(
                        "Error: path traversal outside workspace is not allowed".to_string()
                    );
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err("Error: invalid directory path".to_string());
            }
        }
    }

    let canonical_root = workspace_dir
        .canonicalize()
        .unwrap_or_else(|_| workspace_dir.to_path_buf());
    let candidate = if normalized.as_os_str().is_empty() {
        workspace_dir.to_path_buf()
    } else {
        workspace_dir.join(normalized)
    };

    if candidate.exists() {
        let canonical_candidate = candidate
            .canonicalize()
            .map_err(|_| "Error: invalid directory path".to_string())?;
        if !canonical_candidate.starts_with(&canonical_root) {
            return Err("Error: path traversal outside workspace is not allowed".to_string());
        }
    }

    Ok(candidate)
}

/// Run a shell command inside the workspace directory and capture its output.
fn run_command_in_workspace(program: &str, args: &[&str], cwd: &Path) -> String {
    let result = Command::new(program).args(args).current_dir(cwd).output();

    match result {
        Ok(output) => {
            let mut combined = String::new();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stdout.is_empty() {
                combined.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str("STDERR: ");
                combined.push_str(&stderr);
            }
            if combined.is_empty() {
                combined.push_str("(no output)");
            }
            // Truncate to keep token costs manageable
            truncate_output(&combined, 8_192)
        }
        Err(e) => format!("Error running {}: {}", program, e),
    }
}

/// Truncate a string to at most `max_chars` characters, appending a notice.
fn truncate_output(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_string()
    } else {
        format!("{}\n…(truncated to {} chars)", &s[..max_chars], max_chars)
    }
}

/// Run ASTra semantic search and format the results as a context string.
fn run_astra_search(engine: &SearchEngine, query: &str, top_k: usize) -> String {
    let paths = match engine.search(query, top_k) {
        Ok(p) => p,
        Err(e) => return format!("ASTra search error: {}", e),
    };
    let mut seen = HashSet::new();
    let mut context = String::new();
    for path in paths {
        for node in &path.nodes {
            if !seen.insert(node.symbol_id.clone()) {
                continue;
            }
            context.push_str(&format!(
                "FILE: {}\nSYMBOL: {}\n{}\n\n",
                node.file_path, node.name, node.body
            ));
        }
    }
    if context.is_empty() {
        context.push_str("(no results)");
    }
    truncate_output(&context, 8_192)
}

// ─── Per-case evaluation ──────────────────────────────────────────────────────

/// Run the multi-turn tool-call loop for a single SWE-bench case.
/// Returns a `PredictionRecord` regardless of success/failure.
async fn evaluate_case(
    case: &SweBenchCase,
    engine: &SearchEngine,
    workspace_dir: &Path,
    agent: &MultiTurnAgent,
    max_tool_calls: usize,
    top_k: usize,
) -> PredictionRecord {
    let tools = build_tools();
    let system_prompt = build_system_prompt(max_tool_calls);

    let mut messages: Vec<Message> = vec![
        Message::System {
            content: system_prompt,
            name: None,
        },
        Message::User {
            content: format!("Please fix the following GitHub issue:\n\n{}", case.query),
            name: None,
        },
    ];

    let mut total_tool_calls: usize = 0;
    let mut tool_call_breakdown: HashMap<String, usize> = HashMap::new();
    let mut prompt_tokens: u64 = 0;
    let mut completion_tokens: u64 = 0;
    let mut submitted_patch = String::new();
    let mut completed = false;

    println!(
        "  [{}] Starting turn loop (max_tool_calls={})",
        case.id, max_tool_calls
    );

    loop {
        // Check tool call budget *before* sending to LLM
        if total_tool_calls >= max_tool_calls {
            println!(
                "  [{}] Reached max_tool_calls={} without submit_patch. Marking incomplete.",
                case.id, max_tool_calls
            );
            break;
        }

        let (msg, pt, ct) = match agent.chat(messages.clone(), &tools).await {
            Ok(r) => r,
            Err(e) => {
                println!("  [{}] LLM error: {:?}", case.id, e);
                break;
            }
        };
        prompt_tokens += pt;
        completion_tokens += ct;

        match &msg {
            Message::Assistant {
                tool_calls: Some(tcs),
                ..
            } => {
                // Clone tool calls to process after pushing assistant message
                let tcs = tcs.clone();

                // Push the assistant message (with tool_calls) into history
                messages.push(msg.clone());

                for tc in &tcs {
                    let fn_name = tc.function.name.as_str();

                    // Track call counts
                    total_tool_calls += 1;
                    *tool_call_breakdown.entry(fn_name.to_string()).or_insert(0) += 1;

                    println!(
                        "  [{}] Tool call {}/{}: {}({})",
                        case.id,
                        total_tool_calls,
                        max_tool_calls,
                        fn_name,
                        &tc.function.arguments[..tc.function.arguments.len().min(120)]
                    );

                    // Parse arguments
                    let args: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);

                    // Handle submit_patch specially
                    if fn_name == "submit_patch" {
                        let diff = args
                            .get("diff")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        submitted_patch = diff;
                        completed = true;

                        // Acknowledge to close the conversation properly
                        messages.push(Message::Tool {
                            content: "Patch submitted successfully.".to_string(),
                            tool_call_id: tc.id.clone(),
                        });
                        break; // stop processing further tool calls
                    }

                    // Execute the tool and append the result
                    let output = execute_tool(fn_name, &args, workspace_dir, engine, top_k);
                    messages.push(Message::Tool {
                        content: output,
                        tool_call_id: tc.id.clone(),
                    });

                    // Stop processing more tool calls if we've hit the budget
                    if total_tool_calls >= max_tool_calls {
                        println!(
                            "  [{}] Tool call budget exhausted after {}.",
                            case.id, total_tool_calls
                        );
                        break;
                    }
                }

                if completed {
                    println!("  [{}] submit_patch received. Done.", case.id);
                    break;
                }
            }

            Message::Assistant {
                content: Some(text),
                tool_calls: None,
                ..
            } => {
                // LLM responded with plain text (no tool call). Log and stop.
                println!(
                    "  [{}] LLM returned plain text (no tool call). Stopping.",
                    case.id
                );
                // Try to extract a diff from the plain text as a fallback
                let fallback = extract_diff(text);
                if !fallback.is_empty() {
                    submitted_patch = fallback;
                    completed = true;
                }
                break;
            }

            _ => {
                println!("  [{}] Unexpected message shape. Stopping.", case.id);
                break;
            }
        }
    }

    if !completed {
        println!(
            "  [{}] Evaluation incomplete (patch=empty). tool_calls_used={}",
            case.id, total_tool_calls
        );
    }

    PredictionRecord {
        instance_id: case.id.clone(),
        model_patch: submitted_patch,
        total_tool_calls_used: total_tool_calls,
        tool_call_breakdown,
        prompt_tokens,
        completion_tokens,
    }
}

// ─── Diff extraction fallback ─────────────────────────────────────────────────

fn extract_diff(text: &str) -> String {
    let mut diff = String::new();
    let mut inside = false;
    for line in text.lines() {
        if line.starts_with("```diff") {
            inside = true;
            continue;
        } else if line.starts_with("```") && inside {
            break;
        }
        if inside {
            diff.push_str(line);
            diff.push('\n');
        }
    }
    diff
}

// ─── JSONL/JSON helpers ───────────────────────────────────────────────────────

fn load_cases(path: &Path, max_cases: usize) -> Result<Vec<SweBenchCase>> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read SWE-bench file {}", path.display()))?;
    let mut cases = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let val: Value = serde_json::from_str(line).with_context(|| "invalid JSONL line")?;
        if let Some(case) = case_from_value(&val) {
            cases.push(case);
            if cases.len() >= max_cases {
                break;
            }
        }
    }
    Ok(cases)
}

fn case_from_value(v: &Value) -> Option<SweBenchCase> {
    let id = v
        .get("instance_id")
        .or_else(|| v.get("id"))
        .and_then(Value::as_str)?
        .to_string();
    let repo = v.get("repo").and_then(Value::as_str)?.to_string();
    let query = v
        .get("problem_statement")
        .or_else(|| v.get("query"))
        .and_then(Value::as_str)?
        .to_string();
    Some(SweBenchCase { id, repo, query })
}

// ─── CLI parsing ──────────────────────────────────────────────────────────────

fn parse_args() -> Result<Args> {
    let mut workspace: Option<PathBuf> = None;
    let mut swe_bench_jsonl: Option<PathBuf> = None;
    let mut output = PathBuf::from("benchmarks/reports/predictions.jsonl");
    let mut metrics_output = PathBuf::from("benchmarks/reports/metrics.json");
    let mut max_cases = 50usize;
    let mut top_k = 5usize;
    let mut llmg_model: Option<String> = None;
    let mut llmg_max_tokens = 4096u32;
    let mut max_tool_calls = 10usize;
    let mut concurrency = 8usize;

    let mut args_iter = env::args().skip(1);
    while let Some(arg) = args_iter.next() {
        match arg.as_str() {
            "--workspace" => workspace = args_iter.next().map(PathBuf::from),
            "--swe-bench-jsonl" => swe_bench_jsonl = args_iter.next().map(PathBuf::from),
            "--output" => output = args_iter.next().map(PathBuf::from).unwrap_or(output),
            "--metrics" => {
                metrics_output = args_iter
                    .next()
                    .map(PathBuf::from)
                    .unwrap_or(metrics_output)
            }
            "--max-cases" => {
                max_cases = args_iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --max-cases"))?
                    .parse()
                    .with_context(|| "invalid --max-cases value")?
            }
            "--top-k" => {
                top_k = args_iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --top-k"))?
                    .parse()
                    .with_context(|| "invalid --top-k value")?
            }
            "--llmg-model" => llmg_model = args_iter.next(),
            "--llmg-max-tokens" => {
                llmg_max_tokens = args_iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --llmg-max-tokens"))?
                    .parse()
                    .with_context(|| "invalid --llmg-max-tokens value")?
            }
            "--max-tool-calls" => {
                max_tool_calls = args_iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --max-tool-calls"))?
                    .parse()
                    .with_context(|| "invalid --max-tool-calls value")?
            }
            "--concurrency" => {
                concurrency = args_iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --concurrency"))?
                    .parse()
                    .with_context(|| "invalid --concurrency value")?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown argument: {}", other)),
        }
    }

    let workspace = workspace.ok_or_else(|| anyhow!("--workspace is required"))?;
    let swe_bench_jsonl =
        swe_bench_jsonl.ok_or_else(|| anyhow!("--swe-bench-jsonl is required"))?;
    let llmg_model = llmg_model.ok_or_else(|| anyhow!("--llmg-model is required"))?;

    Ok(Args {
        workspace,
        swe_bench_jsonl,
        output,
        metrics_output,
        max_cases,
        top_k,
        llmg_model,
        llmg_max_tokens,
        max_tool_calls,
        concurrency,
    })
}

fn print_help() {
    println!(
        "Usage:\n\
         cargo run --bin multi_turn_tool_benchmark -- \\\n\
           --workspace <repo_dir> --swe-bench-jsonl <file> --llmg-model <model> [options]\n\n\
         Options:\n\
           --output <file>            predictions output path (default: benchmarks/reports/predictions.jsonl)\n\
           --metrics <file>           metrics summary path   (default: benchmarks/reports/metrics.json)\n\
           --max-cases <n>            max SWE-bench cases to evaluate (default: 50)\n\
           --top-k <n>                ASTra search depth (default: 5)\n\
           --llmg-max-tokens <n>      max completion tokens (default: 4096)\n\
           --max-tool-calls <n>       max tool calls per issue (default: 10)\n\
           --concurrency <n>          parallel evaluation tasks (default: 8)\n"
    );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let cases = load_cases(&args.swe_bench_jsonl, args.max_cases)?;
    if cases.is_empty() {
        return Err(anyhow!(
            "No valid SWE-bench cases found in {}",
            args.swe_bench_jsonl.display()
        ));
    }
    println!("Loaded {} cases.", cases.len());

    let agent =
        Arc::new(MultiTurnAgent::from_env(args.llmg_model.clone(), args.llmg_max_tokens).await?);

    // Group cases by repo so we build/load the Astra index only once per repo.
    let mut cases_by_repo: BTreeMap<String, Vec<SweBenchCase>> = BTreeMap::new();
    for case in cases {
        cases_by_repo
            .entry(case.repo.clone())
            .or_default()
            .push(case);
    }

    // Prepare outputs
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = args.metrics_output.parent() {
        fs::create_dir_all(parent)?;
    }
    let output_file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&args.output)?;
    let writer = Arc::new(Mutex::new(std::io::BufWriter::new(output_file)));

    // Accumulate all records for metrics summary
    let all_records: Arc<Mutex<Vec<PredictionRecord>>> = Arc::new(Mutex::new(Vec::new()));

    for (repo, repo_cases) in cases_by_repo {
        println!("=== Repo: {} ({} cases) ===", repo, repo_cases.len());

        let repo_name = repo.split('/').next_back().unwrap_or(&repo);
        let repo_workspace = args.workspace.join(repo_name);

        if !repo_workspace.exists() {
            eprintln!(
                "Warning: workspace {} does not exist, skipping {} cases for {}",
                repo_workspace.display(),
                repo_cases.len(),
                repo
            );
            continue;
        }

        // Build or load Astra index for this repo
        let repo_config = AstraConfig::new(&repo_workspace);
        let astra_embedder: Box<dyn astra::embeddings::Embedder> =
            astra::embeddings::build_embedder(repo_config.embedding_provider.as_str())?;

        let (graph, vector_store) = if astra::storage::has_persisted_data(&repo_config) {
            println!("Loading existing Astra index for {}…", repo_name);
            let mut g = astra::storage::load_graph(&repo_config)?;
            g.rebuild_after_deserialize();
            let mut v = astra::storage::load_vector_store(&repo_config)?;
            v.rebuild_index();
            (g, v)
        } else {
            println!("Indexing {} for Astra…", repo_name);
            let indexed = indexer::index_workspace(&repo_config, astra_embedder.as_ref())
                .with_context(|| {
                    format!("failed to index workspace {}", repo_workspace.display())
                })?;
            (indexed.graph, indexed.vector_store)
        };

        let engine = Arc::new(SearchEngine::new(graph, vector_store, astra_embedder));

        // Evaluate cases in parallel within this repo
        let semaphore = Arc::new(Semaphore::new(args.concurrency));
        let mut handles = Vec::new();

        for case in repo_cases {
            let case = case.clone();
            let engine = engine.clone();
            let agent = agent.clone();
            let writer = writer.clone();
            let all_records = all_records.clone();
            let workspace_dir = repo_workspace.clone();
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let max_tool_calls = args.max_tool_calls;
            let top_k = args.top_k;

            handles.push(tokio::spawn(async move {
                let _permit = permit;
                let record = evaluate_case(
                    &case,
                    &engine,
                    &workspace_dir,
                    &agent,
                    max_tool_calls,
                    top_k,
                )
                .await;

                let jsonl = serde_json::to_string(&record).unwrap_or_default();
                {
                    let mut w = writer.lock().await;
                    let _ = writeln!(w, "{}", jsonl);
                    let _ = w.flush();
                }
                {
                    let mut records = all_records.lock().await;
                    records.push(record);
                }
            }));
        }

        for handle in handles {
            if let Err(e) = handle.await {
                eprintln!("Task panicked: {:?}", e);
            }
        }
    }

    // Write metrics summary
    {
        let records = all_records.lock().await;
        write_metrics_summary(&records, &args.metrics_output)?;
    }

    println!(
        "\nDone. Predictions → {}  |  Metrics → {}",
        args.output.display(),
        args.metrics_output.display()
    );
    Ok(())
}

fn write_metrics_summary(records: &[PredictionRecord], path: &Path) -> Result<()> {
    let total = records.len();
    let completed = records.iter().filter(|r| !r.model_patch.is_empty()).count();
    let failed = total - completed;

    let avg_tool_calls = if total > 0 {
        records
            .iter()
            .map(|r| r.total_tool_calls_used as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };
    let avg_prompt = if total > 0 {
        records.iter().map(|r| r.prompt_tokens as f64).sum::<f64>() / total as f64
    } else {
        0.0
    };
    let avg_completion = if total > 0 {
        records
            .iter()
            .map(|r| r.completion_tokens as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };

    let mut tool_call_totals: HashMap<String, usize> = HashMap::new();
    for r in records {
        for (tool, count) in &r.tool_call_breakdown {
            *tool_call_totals.entry(tool.clone()).or_insert(0) += count;
        }
    }

    let summary = MetricsSummary {
        total_cases: total,
        completed,
        failed_or_incomplete: failed,
        avg_tool_calls,
        avg_prompt_tokens: avg_prompt,
        avg_completion_tokens: avg_completion,
        tool_call_totals,
    };

    let json = serde_json::to_string_pretty(&summary)?;
    fs::write(path, json)?;
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_diff_from_code_block() {
        let text = "Here is my fix:\n```diff\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-bad\n+good\n```\nDone.";
        let diff = extract_diff(text);
        assert_eq!(
            diff,
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-bad\n+good\n"
        );
    }

    #[test]
    fn test_extract_diff_no_block() {
        let text = "No diff here.";
        let diff = extract_diff(text);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_truncate_output() {
        let s = "a".repeat(20_000);
        let t = truncate_output(&s, 8_192);
        assert!(t.len() > 8_192); // includes the notice suffix
        assert!(t.starts_with(&"a".repeat(8_192)));
        assert!(t.contains("truncated"));
    }

    #[test]
    fn test_case_from_value_valid() {
        let v = serde_json::json!({
            "instance_id": "django__django-12345",
            "repo": "django/django",
            "problem_statement": "Fix the bug in QuerySet."
        });
        let case = case_from_value(&v).unwrap();
        assert_eq!(case.id, "django__django-12345");
        assert_eq!(case.repo, "django/django");
        assert_eq!(case.query, "Fix the bug in QuerySet.");
    }

    #[test]
    fn test_case_from_value_missing_repo() {
        let v = serde_json::json!({
            "instance_id": "foo",
            "problem_statement": "some issue"
        });
        assert!(case_from_value(&v).is_none());
    }

    #[test]
    fn test_build_system_prompt_contains_limit() {
        let prompt = build_system_prompt(3);
        assert!(prompt.contains("3 tool call(s)"));
    }

    #[test]
    fn test_build_tools_names() {
        let tools = build_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"search_astra"));
        assert!(names.contains(&"run_grep"));
        assert!(names.contains(&"run_ripgrep"));
        assert!(names.contains(&"submit_patch"));
    }

    #[test]
    fn test_write_metrics_summary() {
        let records = vec![
            PredictionRecord {
                instance_id: "a".to_string(),
                model_patch: "diff".to_string(),
                total_tool_calls_used: 3,
                tool_call_breakdown: [("search_astra".to_string(), 2), ("run_grep".to_string(), 1)]
                    .into_iter()
                    .collect(),
                prompt_tokens: 100,
                completion_tokens: 50,
            },
            PredictionRecord {
                instance_id: "b".to_string(),
                model_patch: String::new(),
                total_tool_calls_used: 10,
                tool_call_breakdown: [("run_ripgrep".to_string(), 10)].into_iter().collect(),
                prompt_tokens: 200,
                completion_tokens: 100,
            },
        ];

        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_metrics_summary(&records, tmp.path()).unwrap();
        let content = fs::read_to_string(tmp.path()).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["total_cases"], 2);
        assert_eq!(v["completed"], 1);
        assert_eq!(v["failed_or_incomplete"], 1);
    }

    #[test]
    fn test_resolve_tool_dir_rejects_absolute_paths() {
        let workspace = tempfile::tempdir().unwrap();
        let result = resolve_tool_dir("/etc", workspace.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_tool_dir_rejects_parent_escape() {
        let workspace = tempfile::tempdir().unwrap();
        let result = resolve_tool_dir("../../etc", workspace.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_tool_dir_allows_in_workspace_paths() {
        let workspace = tempfile::tempdir().unwrap();
        let nested = workspace.path().join("src/subdir");
        fs::create_dir_all(&nested).unwrap();
        let resolved = resolve_tool_dir("src/subdir", workspace.path()).unwrap();
        assert_eq!(resolved, nested);
    }
}
