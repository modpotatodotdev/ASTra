use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use astra::config::AstraConfig;
use astra::embeddings::{cosine_similarity, Embedder};
use astra::indexer;
use astra::search::SearchEngine;
use llmg_core::provider::{Provider, ProviderRegistry, RoutingProvider};
use llmg_core::types::{ChatCompletionRequest, Message};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum Method {
    Astra,
    Grep,
    Ripgrep,
    TraditionalRag,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RetrievalResult {
    context: String,
    files: Vec<String>,
    retrieval_ms: f64,
    skeleton_tokens: u64,
    body_tokens: u64,
}

#[derive(Clone)]
struct Args {
    workspace: PathBuf,
    swe_bench_jsonl: PathBuf,
    output: PathBuf,
    max_cases: usize,
    top_k: usize,
    llmg_model: String,
    llmg_max_tokens: u32,
    concurrency: usize,
    method: Method,
}

#[derive(Debug, Clone)]
struct SweBenchCase {
    id: String,
    repo: String,
    query: String,
}

#[derive(Debug, Serialize)]
struct PredictionRecord {
    instance_id: String,
    model_name_or_path: String,
    model_patch: String,
    method: Method,
}

#[derive(Debug)]
struct LlmgPatchGenerator {
    client: RoutingProvider,
    model: String,
    max_tokens: u32,
}

impl LlmgPatchGenerator {
    async fn from_env(model: String, max_tokens: u32) -> Result<Self> {
        let mut registry = ProviderRegistry::new();
        llmg_providers::utils::register_all_from_env(&mut registry).await;

        if registry.list().is_empty() {
            return Err(anyhow::anyhow!(
                "No LLMG providers are configured via environment variables"
            ));
        }

        let client = RoutingProvider::new(registry);

        Ok(Self {
            client,
            model,
            max_tokens,
        })
    }

    async fn generate_patch(&self, query: &str, context: &str) -> Result<String> {
        for attempt in 0..5 {
            let request = ChatCompletionRequest {
                model: self.model.clone(),
                messages: vec![
                    Message::System {
                        content: "You are an expert software engineer resolving GitHub issues. You are given an issue description and a set of retrieved code files and snippets that are likely relevant to fixing the issue.\n\nYour task is to write a patch to resolve the issue. You MUST output your patch inside a markdown block with the language `diff`, like so:\n```diff\n--- a/src/main.py\n+++ b/src/main.py\n@@ -10,5 +10,6 @@\n def example():\n-    return False\n+    print(\"fixed!\")\n+    return True\n```\nMake sure your diff is a unified diff applied from the root of the repository (with a/ and b/ prefixes). Do strictly only provide the diff to fix the bug, do not provide any explanations.".to_string(),
                        name: None,
                    },
                    Message::User {
                        content: format!(
                            "Issue Description:\n{}\n\nRetrieved context:\n{}\n\nPlease provide the diff to resolve this issue.",
                            query, context
                        ),
                        name: None,
                    },
                ],
                temperature: Some(0.0),
                max_tokens: Some(self.max_tokens),
                stream: Some(false),
                top_p: None,
                frequency_penalty: None,
                presence_penalty: None,
                stop: None,
                user: Some("ASTra-zero-shot-patch-benchmark".to_string()),
                tools: None,
                tool_choice: None,
            };

            match self.client.chat_completion(request).await {
                Ok(response) => {
                    let completion = response
                        .choices
                        .first()
                        .and_then(|choice| match &choice.message {
                            Message::Assistant { content, .. } => content.clone(),
                            _ => None,
                        })
                        .unwrap_or_default();
                    return Ok(extract_diff(&completion));
                }
                Err(e) => {
                    println!("LLM API Request failed on attempt {}: {:?}", attempt + 1, e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(2_u64.pow(attempt))).await;
                }
            }
        }

        println!("All LLM retry attempts failed. Returning empty patch.");
        Ok(String::new())
    }
}

fn extract_diff(llm_output: &str) -> String {
    let mut diff = String::new();
    let mut inside_diff = false;

    for line in llm_output.lines() {
        if line.starts_with("```diff") {
            inside_diff = true;
            continue;
        } else if line.starts_with("```") && inside_diff {
            // Assuming at most one diff block
            break;
        }

        if inside_diff {
            diff.push_str(line);
            diff.push('\n');
        }
    }

    // fallback, return entire output if no diff blocks found
    if diff.is_empty() {
        return llm_output.to_string();
    }

    diff
}

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

    let llmg_generator =
        Arc::new(LlmgPatchGenerator::from_env(args.llmg_model, args.llmg_max_tokens).await?);

    // Group cases by repo
    let mut cases_by_repo: BTreeMap<String, Vec<&SweBenchCase>> = BTreeMap::new();
    for case in &cases {
        cases_by_repo
            .entry(case.repo.clone())
            .or_default()
            .push(case);
    }

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let output_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)?;
    let mut writer = std::io::BufWriter::new(output_file);

    // Evaluate per repo
    for (repo, repo_cases) in cases_by_repo {
        println!("Evaluating {} cases for repo {}", repo_cases.len(), repo);

        let repo_name = repo.split('/').next_back().unwrap_or(&repo);
        let repo_workspace = args.workspace.join(repo_name);

        if !repo_workspace.exists() {
            eprintln!(
                "Warning: workspace {} does not exist, skipping cases for {}",
                repo_workspace.display(),
                repo
            );
            continue;
        }

        let repo_config = AstraConfig::new(&repo_workspace);

        let astra_embedder = astra::embeddings::build_embedder(repo_config.embedding_provider.as_str())?;

        let (graph, vector_store) = if astra::storage::has_persisted_data(&repo_config) {
            println!("Loading existing ASTra index for {}...", repo_name);
            let mut g = astra::storage::load_graph(&repo_config)?;
            g.rebuild_after_deserialize();
            let mut v = astra::storage::load_vector_store(&repo_config)?;
            v.rebuild_index();
            (g, v)
        } else {
            println!("Indexing {} for ASTra...", repo_name);
            let indexed = indexer::index_workspace(&repo_config, astra_embedder.as_ref())
                .with_context(|| {
                    format!("failed to index workspace {}", repo_workspace.display())
                })?;
            (indexed.graph, indexed.vector_store)
        };

        let astra_engine = Arc::new(SearchEngine::new(graph, vector_store, astra_embedder));

        println!(
            "Loading or building Traditional RAG index for {}...",
            repo_name
        );
        let rag_index = Arc::new(TraditionalRagIndex::load_or_build(
            &repo_workspace,
            &repo_config,
        )?);

        let semaphore = Arc::new(tokio::sync::Semaphore::new(args.concurrency));
        let mut handles = Vec::new();
        let method = args.method;
        let repo_workspace = Arc::new(repo_workspace);

        for case in repo_cases {
            let case = case.clone();
            let astra_engine = astra_engine.clone();
            let rag_index = rag_index.clone();
            let repo_workspace = repo_workspace.clone();
            let llmg_generator = llmg_generator.clone();
            let top_k = args.top_k;
            let permit = semaphore.clone().acquire_owned().await.unwrap();

            handles.push(tokio::spawn(async move {
                let _permit = permit;

                let retrieved =
                    match method {
                        Method::Astra => run_astra(&astra_engine, &case.query, top_k),
                        Method::Grep => run_grep(&repo_workspace, &case.query, top_k)
                            .unwrap_or_else(|_| RetrievalResult {
                                context: String::new(),
                                files: vec![],
                                retrieval_ms: 0.0,
                                skeleton_tokens: 0,
                                body_tokens: 0,
                            }),
                        Method::Ripgrep => run_ripgrep(&repo_workspace, &case.query, top_k)
                            .unwrap_or_else(|_| RetrievalResult {
                                context: String::new(),
                                files: vec![],
                                retrieval_ms: 0.0,
                                skeleton_tokens: 0,
                                body_tokens: 0,
                            }),
                        Method::TraditionalRag => rag_index.search(&case.query, top_k),
                    };

                let patch = llmg_generator
                    .generate_patch(&case.query, &retrieved.context)
                    .await?;

                let record = PredictionRecord {
                    instance_id: case.id.clone(),
                    model_name_or_path: llmg_generator.model.clone(),
                    model_patch: patch,
                    method,
                };

                let jsonl = serde_json::to_string(&record).unwrap();
                Ok::<String, anyhow::Error>(jsonl)
            }));
        }

        for handle in handles {
            if let Ok(Ok(jsonl)) = handle.await {
                use std::io::Write;
                writeln!(writer, "{}", jsonl)?;
                writer.flush()?;
            }
        }
    }

    println!("Benchmark report written to {}", args.output.display());
    Ok(())
}

const CONTEXT_TOKEN_BUDGET: u64 = 16_384;

fn run_astra(engine: &SearchEngine, query: &str, top_k: usize) -> RetrievalResult {
    let start = Instant::now();
    let paths = engine.search(query, top_k).unwrap();
    let mut seen_symbols = HashSet::new();
    let mut context = String::new();
    let files = Vec::new();
    let mut context_body_tokens: u64 = 0;
    let mut context_skeleton_tokens: u64 = 0;
    for path in paths {
        if path.nodes.is_empty() {
            continue;
        }
        let mut final_nodes = Vec::new();

        if !path.nodes.is_empty() {
            final_nodes.push(path.nodes.first().unwrap());
            if path.nodes.len() > 2 {
                let middle_nodes = &path.nodes[1..path.nodes.len() - 1];
                if let Some(best_mid) = middle_nodes
                    .iter()
                    .max_by(|a, b| a.relevance.partial_cmp(&b.relevance).unwrap())
                {
                    final_nodes.push(best_mid);
                }
            }
            if path.nodes.len() > 1 {
                final_nodes.push(path.nodes.last().unwrap());
            }
        }

        let last_idx = final_nodes.len().saturating_sub(1);
        for (idx, node) in final_nodes.into_iter().enumerate() {
            if !seen_symbols.insert(node.symbol_id.clone()) {
                continue;
            }
            if context_body_tokens + context_skeleton_tokens >= CONTEXT_TOKEN_BUDGET {
                continue;
            }
            let is_final_target = idx == last_idx;
            let payload = if is_final_target {
                node.body.clone()
            } else {
                node.skeleton_context()
            };
            let entry = format!(
                "FILE: {}\nSYMBOL: {}\nCONTEXT_MODE: {}\n{}\n\n",
                node.file_path,
                node.name,
                if is_final_target { "full" } else { "skeleton" },
                payload
            );
            let tokens = estimate_tokens(&entry);
            if is_final_target {
                context_body_tokens += tokens;
            } else {
                context_skeleton_tokens += tokens;
            }
            context.push_str(&entry);
        }
    }
    RetrievalResult {
        context,
        files,
        retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
        skeleton_tokens: context_skeleton_tokens,
        body_tokens: context_body_tokens,
    }
}

fn estimate_tokens(text: &str) -> u64 {
    text.len() as u64 / 3
}

fn load_cases(path: &Path, max_cases: usize) -> Result<Vec<SweBenchCase>> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read SWE-bench file {}", path.display()))?;
    let mut cases = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).with_context(|| "invalid JSONL line")?;
        if let Some(case) = case_from_value(&value) {
            cases.push(case);
            if cases.len() >= max_cases {
                break;
            }
        }
    }
    Ok(cases)
}

fn case_from_value(value: &Value) -> Option<SweBenchCase> {
    let id = value
        .get("instance_id")
        .or_else(|| value.get("id"))
        .and_then(Value::as_str)?
        .to_string();
    let repo = value.get("repo").and_then(Value::as_str)?.to_string();
    let query = value
        .get("problem_statement")
        .or_else(|| value.get("query"))
        .and_then(Value::as_str)?
        .to_string();
    Some(SweBenchCase { id, repo, query })
}

fn parse_args() -> Result<Args> {
    let mut workspace: Option<PathBuf> = None;
    let mut swe_bench_jsonl: Option<PathBuf> = None;
    let mut output = PathBuf::from("benchmarks/reports/predictions.jsonl");
    let mut max_cases = 50usize;
    let mut top_k = 10usize;
    let mut llmg_model: Option<String> = None;
    let mut llmg_max_tokens = 4096u32;
    let mut concurrency = 8usize;
    let mut method = Method::Astra;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workspace" => workspace = args.next().map(PathBuf::from),
            "--swe-bench-jsonl" => swe_bench_jsonl = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from).unwrap_or(output),
            "--max-cases" => {
                max_cases = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --max-cases"))?
                    .parse()
                    .with_context(|| "invalid --max-cases value")?
            }
            "--top-k" => {
                top_k = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --top-k"))?
                    .parse()
                    .with_context(|| "invalid --top-k value")?
            }
            "--llmg-model" => llmg_model = args.next(),
            "--llmg-max-tokens" => {
                llmg_max_tokens = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --llmg-max-tokens"))?
                    .parse()
                    .with_context(|| "invalid --llmg-max-tokens value")?
            }
            "--concurrency" => {
                concurrency = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --concurrency"))?
                    .parse()
                    .with_context(|| "invalid --concurrency value")?
            }
            "--method" => {
                let m = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --method"))?;
                method = match m.as_str() {
                    "astra" => Method::Astra,
                    "grep" => Method::Grep,
                    "ripgrep" => Method::Ripgrep,
                    "traditional_rag" => Method::TraditionalRag,
                    _ => return Err(anyhow!("invalid method: {}", m)),
                };
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
        max_cases,
        top_k,
        llmg_model,
        llmg_max_tokens,
        concurrency,
        method,
    })
}

fn print_help() {
    println!(
        "Usage:\n  cargo run --bin zero_shot_patch_benchmark -- --workspace <repo> --swe-bench-jsonl <file> --llmg-model <model> [options]\n\nOptions:\n  --output <file>            Output JSON report path (default: benchmarks/reports/predictions.jsonl)\n  --max-cases <n>            Number of SWE-bench cases to evaluate (default: 50)\n  --top-k <n>                Retrieval depth per method (default: 10)\n  --llmg-max-tokens <n>      Max completion tokens for LLMG call (default: 4096)\n  --concurrency <n>          Number of concurrent evaluations (default: 8)\n  --method <method>          Retrieval method (ASTra|grep|ripgrep|traditional_rag, default: ASTra)\n"
    );
}

#[derive(Clone, Serialize, Deserialize)]
struct RagChunk {
    file_path: String,
    content: String,
    embedding: Vec<f32>,
}

struct TraditionalRagIndex {
    embedder: Box<dyn Embedder>,
    chunks: Vec<RagChunk>,
}

impl TraditionalRagIndex {
    fn cache_is_valid(workspace: &Path, chunks: &[RagChunk], expected_dim: usize) -> bool {
        chunks.iter().all(|chunk| {
            !chunk.content.is_empty()
                && chunk.embedding.len() == expected_dim
                && workspace.join(&chunk.file_path).exists()
        })
    }

    fn build(workspace: &Path, config: &AstraConfig) -> Result<Self> {
        let embedder = astra::embeddings::build_embedder(config.embedding_provider.as_str())?;
        let files = indexer::collect_files(workspace, &config.extensions)?;
        let mut chunks = Vec::new();

        for file in files {
            let relative = Path::new(&file)
                .strip_prefix(workspace)
                .unwrap_or(Path::new(&file))
                .to_string_lossy()
                .to_string();
            let content = match fs::read_to_string(&file) {
                Ok(c) => c,
                Err(_) => continue,
            };
            for chunk in chunk_content(&content, 40, 20) {
                let embedding = embedder.embed(&chunk).unwrap();
                chunks.push(RagChunk {
                    file_path: relative.clone(),
                    content: chunk,
                    embedding,
                });
            }
        }

        Ok(Self { embedder, chunks })
    }

    fn load_or_build(workspace: &Path, config: &AstraConfig) -> Result<Self> {
        let cache_path = config.data_dir.join("traditional_rag.bin");
        if cache_path.exists() {
            println!("Loading existing Traditional RAG index...");
            if let Ok(data) = fs::read(&cache_path) {
                if let Ok(chunks) = bincode::deserialize::<Vec<RagChunk>>(&data) {
                    let embedder = astra::embeddings::build_embedder(config.embedding_provider.as_str())?;
                    if Self::cache_is_valid(workspace, &chunks, embedder.dim()) {
                        return Ok(Self { embedder, chunks });
                    }
                    println!("Cached Traditional RAG index failed integrity checks; rebuilding...");
                }
            }
        }

        println!(
            "Building Traditional RAG index for {}...",
            workspace.display()
        );
        let index = Self::build(workspace, config)?;
        if let Some(parent) = cache_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(data) = bincode::serialize(&index.chunks) {
            let _ = fs::write(&cache_path, data);
        }
        Ok(index)
    }

    fn search(&self, query: &str, top_k: usize) -> RetrievalResult {
        let start = Instant::now();
        let query_embedding = self.embedder.embed(query).unwrap();
        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| (i, cosine_similarity(&query_embedding, &chunk.embedding)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let mut context_parts = Vec::new();
        let mut seen = HashSet::new();
        let mut files = Vec::new();
        for (index, score) in scored {
            let chunk = &self.chunks[index];
            let normalized = normalize_match_path(&chunk.file_path);
            if seen.insert(normalized.clone()) {
                files.push(normalized);
            }
            context_parts.push(format!(
                "FILE: {}\nSCORE: {:.4}\n{}",
                chunk.file_path, score, chunk.content
            ));
        }

        let context = context_parts.join("\n\n---\n\n");
        RetrievalResult {
            skeleton_tokens: 0,
            body_tokens: estimate_tokens(&context),
            context,
            files,
            retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

fn run_grep(workspace: &Path, query: &str, top_k: usize) -> Result<RetrievalResult> {
    let start = Instant::now();
    let needles = build_text_needles(query);
    if needles.is_empty() {
        return Ok(RetrievalResult {
            context: String::new(),
            files: Vec::new(),
            retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
            skeleton_tokens: 0,
            body_tokens: 0,
        });
    }
    let mut cmd = Command::new("grep");
    cmd.arg("-R")
        .arg("-l")
        .arg("-I")
        .arg("-F")
        .arg("--exclude-dir=.git")
        .arg("--exclude-dir=target")
        .arg("--exclude-dir=node_modules")
        .arg("--exclude-dir=.folder");
    for needle in &needles {
        cmd.arg("-e").arg(needle);
    }
    cmd.arg(workspace);
    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return run_builtin_text_search(workspace, query, top_k);
        }
        Err(e) => return Err(e).with_context(|| "failed to execute grep"),
    };
    if !output.status.success() {
        if output.status.code() == Some(1) {
            return Ok(RetrievalResult {
                context: String::new(),
                files: Vec::new(),
                retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
                skeleton_tokens: 0,
                body_tokens: 0,
            });
        }
        return run_builtin_text_search(workspace, query, top_k);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for line in stdout.lines() {
        let rel = path_relative_to(workspace, line.trim());
        if !rel.is_empty() && seen.insert(rel.clone()) {
            files.push(rel);
            if files.len() >= top_k {
                break;
            }
        }
    }
    let context = files
        .iter()
        .map(|f| format!("FILE: {}", f))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(RetrievalResult {
        skeleton_tokens: 0,
        body_tokens: estimate_tokens(&context),
        context,
        files,
        retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

fn run_ripgrep(workspace: &Path, query: &str, top_k: usize) -> Result<RetrievalResult> {
    let start = Instant::now();
    let needles = build_text_needles(query);
    if needles.is_empty() {
        return Ok(RetrievalResult {
            context: String::new(),
            files: Vec::new(),
            retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
            skeleton_tokens: 0,
            body_tokens: 0,
        });
    }
    let mut cmd = Command::new("rg");
    cmd.arg("--fixed-strings").arg("--files-with-matches");
    for needle in &needles {
        cmd.arg("-e").arg(needle);
    }
    cmd.arg(workspace);
    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return run_builtin_text_search(workspace, query, top_k);
        }
        Err(e) => return Err(e).with_context(|| "failed to execute ripgrep"),
    };
    if !output.status.success() {
        if output.status.code() == Some(1) {
            return Ok(RetrievalResult {
                context: String::new(),
                files: Vec::new(),
                retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
                skeleton_tokens: 0,
                body_tokens: 0,
            });
        }
        return run_builtin_text_search(workspace, query, top_k);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for line in stdout.lines() {
        let rel = path_relative_to(workspace, line.trim());
        if !rel.is_empty() && seen.insert(rel.clone()) {
            files.push(rel);
            if files.len() >= top_k {
                break;
            }
        }
    }
    let context = files
        .iter()
        .map(|f| format!("FILE: {}", f))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(RetrievalResult {
        skeleton_tokens: 0,
        body_tokens: estimate_tokens(&context),
        context,
        files,
        retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

fn path_relative_to(workspace: &Path, file: &str) -> String {
    let relative = Path::new(file)
        .strip_prefix(workspace)
        .unwrap_or(Path::new(file))
        .to_string_lossy()
        .to_string();
    normalize_match_path(&relative)
}

fn normalize_match_path(path: &str) -> String {
    let mut normalized = path.trim().replace('\\', "/");
    while let Some(stripped) = normalized.strip_prefix("./") {
        normalized = stripped.to_string();
    }
    while let Some(stripped) = normalized.strip_prefix('/') {
        normalized = stripped.to_string();
    }
    normalized
}

fn sanitize_text_query(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn build_text_needles(query: &str) -> Vec<String> {
    let sanitized = sanitize_text_query(query);
    if sanitized.is_empty() {
        return Vec::new();
    }

    let mut needles: Vec<String> = Vec::new();
    let mut seen = HashSet::new();

    let dotted_re =
        regex::Regex::new(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*){1,}").unwrap();
    for m in dotted_re.find_iter(&sanitized) {
        let full = m.as_str();
        if let Some(last) = full.rsplit('.').next() {
            if last.len() >= 3 {
                try_add_needle(&mut needles, &mut seen, last.to_string());
            }
        }
        try_add_needle(&mut needles, &mut seen, full.to_string());
        if needles.len() >= 20 {
            break;
        }
    }

    let ident_re = regex::Regex::new(
        r"[A-Z][a-zA-Z0-9]{2,}(?:_[a-zA-Z0-9]+)*|[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}",
    )
    .unwrap();
    for m in ident_re.find_iter(&sanitized) {
        let tok = m.as_str();
        if is_stop_word(tok) {
            continue;
        }
        try_add_needle(&mut needles, &mut seen, tok.to_string());
        if needles.len() >= 20 {
            break;
        }
    }

    let backtick_re = regex::Regex::new(r"`([^`]{3,})`").unwrap();
    for cap in backtick_re.captures_iter(&sanitized) {
        if let Some(inner) = cap.get(1) {
            let cleaned = inner.as_str().trim_end_matches("()");
            try_add_needle(&mut needles, &mut seen, cleaned.to_string());
            if needles.len() >= 20 {
                break;
            }
        }
    }

    if needles.len() < 5 {
        let normalized = sanitized.to_ascii_lowercase();
        for token in normalized
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-')
            .filter(|token| token.len() >= 5)
        {
            if is_stop_word(token) {
                continue;
            }
            try_add_needle(&mut needles, &mut seen, token.to_string());
            if needles.len() >= 15 {
                break;
            }
        }
    }

    needles
}

fn try_add_needle(needles: &mut Vec<String>, seen: &mut HashSet<String>, s: String) {
    let lower = s.to_ascii_lowercase();
    if lower.len() >= 3 && seen.insert(lower) {
        needles.push(s);
    }
}

fn is_stop_word(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "the"
            | "this"
            | "that"
            | "with"
            | "from"
            | "have"
            | "been"
            | "were"
            | "will"
            | "would"
            | "could"
            | "should"
            | "there"
            | "their"
            | "which"
            | "when"
            | "where"
            | "what"
            | "about"
            | "does"
            | "into"
            | "some"
            | "other"
            | "than"
            | "then"
            | "also"
            | "just"
            | "only"
            | "more"
            | "like"
            | "each"
            | "make"
            | "made"
            | "after"
            | "before"
            | "using"
            | "used"
            | "because"
            | "since"
            | "being"
            | "here"
            | "very"
            | "description"
            | "result"
            | "results"
            | "expected"
            | "actual"
            | "following"
            | "example"
            | "issue"
            | "error"
            | "problem"
            | "warning"
            | "note"
            | "above"
            | "below"
            | "case"
            | "added"
            | "think"
            | "same"
            | "need"
            | "work"
            | "works"
    )
}

fn run_builtin_text_search(workspace: &Path, query: &str, top_k: usize) -> Result<RetrievalResult> {
    let start = Instant::now();
    let config = AstraConfig::new(workspace);
    let files = indexer::collect_files(workspace, &config.extensions)?;

    let mut lines = Vec::new();
    let mut seen = HashSet::new();
    let mut matched_files = Vec::new();
    let needles = build_text_needles(query);
    if needles.is_empty() {
        return Ok(RetrievalResult {
            context: String::new(),
            files: Vec::new(),
            retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
            skeleton_tokens: 0,
            body_tokens: 0,
        });
    }
    let max_lines = top_k.saturating_mul(5);

    for file in files {
        if lines.len() >= max_lines {
            break;
        }
        let relative = path_relative_to(workspace, &file);
        let content = match fs::read_to_string(&file) {
            Ok(content) => content,
            Err(_) => continue,
        };
        for (index, line) in content.lines().enumerate() {
            let lower_line = line.to_ascii_lowercase();
            if needles.iter().any(|needle| lower_line.contains(needle)) {
                lines.push(format!("{}:{}:{}", relative, index + 1, line));
                if seen.insert(relative.clone()) {
                    matched_files.push(relative.clone());
                }
                if lines.len() >= max_lines {
                    break;
                }
            }
        }
    }

    let context = lines.join("\n");
    Ok(RetrievalResult {
        skeleton_tokens: 0,
        body_tokens: estimate_tokens(&context),
        context,
        files: matched_files,
        retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

fn chunk_content(content: &str, chunk_lines: usize, stride: usize) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let end = usize::min(start + chunk_lines, lines.len());
        let chunk = lines[start..end].join("\n");
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
        if end == lines.len() {
            break;
        }
        start = start.saturating_add(stride.max(1));
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::extract_diff;

    #[test]
    fn test_extract_diff() {
        let text = "Here is the patch:\n```diff\n--- a/test\n+++ b/test\n@@ -1 +1 @@\n-a\n+b\n```\nIt should fix the bug.";
        let diff = extract_diff(text);
        assert_eq!(diff, "--- a/test\n+++ b/test\n@@ -1 +1 @@\n-a\n+b\n");
    }

    #[test]
    fn test_extract_diff_fallback() {
        let text = "--- a/test\n+++ b/test\n@@ -1 +1 @@\n-a\n+b\n";
        let diff = extract_diff(text);
        assert_eq!(diff, text);
    }
}
