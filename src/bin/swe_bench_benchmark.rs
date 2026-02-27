use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use astra::config::AstraConfig;
use astra::embeddings::{cosine_similarity, Embedder};
use astra::indexer;
use astra::search::SearchEngine;
use llmg_core::provider::{Provider, ProviderRegistry, RoutingProvider};
use llmg_core::types::{ChatCompletionRequest, Message};
use serde::Serialize;
use serde_json::Value;

#[derive(Clone)]
struct Args {
    workspace: PathBuf,
    swe_bench_jsonl: PathBuf,
    output: PathBuf,
    max_cases: usize,
    top_k: usize,
    llmg_model: Option<String>,
    llmg_max_tokens: u32,
}

#[derive(Debug, Clone)]
struct SweBenchCase {
    id: String,
    repo: String,
    query: String,
    oracle_files: HashSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum Method {
    Astra,
    Grep,
    Ripgrep,
    TraditionalRag,
}

#[derive(Debug, Clone)]
struct RetrievalResult {
    context: String,
    files: Vec<String>,
    retrieval_ms: f64,
    skeleton_tokens: u64,
    body_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkRecord {
    case_id: String,
    method: Method,
    retrieval_ms: f64,
    total_ms: f64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    skeleton_tokens: u64,
    body_tokens: u64,
    oracle_hit: bool,
    recall: f64,
    retrieved_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct MethodSummary {
    runs: usize,
    retrieval_avg_ms: f64,
    retrieval_median_ms: f64,
    retrieval_p95_ms: f64,
    total_avg_ms: f64,
    total_p95_ms: f64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    total_skeleton_tokens: u64,
    total_body_tokens: u64,
    oracle_hit_rate: f64,
    recall_at_k: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    metadata: BenchmarkMetadata,
    summary: BTreeMap<Method, MethodSummary>,
    records: Vec<BenchmarkRecord>,
}

#[derive(Debug, Serialize)]
struct BenchmarkMetadata {
    workspace: String,
    swe_bench_jsonl: String,
    timestamp_unix_secs: u64,
    cases_evaluated: usize,
    top_k: usize,
    llmg_model: Option<String>,
    methodology: String,
}

#[derive(Debug)]
struct LlmgEvaluator {
    client: RoutingProvider,
    model: String,
    max_tokens: u32,
}

impl LlmgEvaluator {
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
                // For other providers, register non-interactive ones only
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

    async fn evaluate(&self, query: &str, context: &str) -> Result<(u64, u64, u64)> {
        for attempt in 0..5 {
            let request = ChatCompletionRequest {
                model: self.model.clone(),
                messages: vec![
                    Message::System {
                        content: "You are solving SWE-bench issues. Use only the provided repository context.".to_string(),
                        name: None,
                    },
                    Message::User {
                        content: format!(
                            "Issue:\n{}\n\nRetrieved context:\n{}\n\nProvide a concise implementation plan.",
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
                user: Some("astra-swe-bench-benchmark".to_string()),
                tools: None,
                tool_choice: None,
            };

            match self.client.chat_completion(request).await {
                Ok(response) => {
                    if let Some(usage) = response.usage {
                        return Ok((
                            usage.prompt_tokens as u64,
                            usage.completion_tokens as u64,
                            usage.total_tokens as u64,
                        ));
                    }

                    let completion = response
                        .choices
                        .first()
                        .and_then(|choice| match &choice.message {
                            Message::Assistant { content, .. } => content.clone(),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let prompt_tokens = estimate_tokens(context) + estimate_tokens(query);
                    let completion_tokens = estimate_tokens(&completion);
                    return Ok((
                        prompt_tokens,
                        completion_tokens,
                        prompt_tokens + completion_tokens,
                    ));
                }
                Err(e) => {
                    println!("LLM API Request failed on attempt {}: {:?}", attempt + 1, e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(2_u64.pow(attempt))).await;
                }
            }
        }

        println!("All LLM retry attempts failed. Returning 0 tokens.");
        Ok((0, 0, 0))
    }
}

use serde::Deserialize;

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

    let llmg = match args.llmg_model.clone() {
        Some(model) => Some(Arc::new(
            LlmgEvaluator::from_env(model, args.llmg_max_tokens).await?,
        )),
        None => None,
    };

    let mut records = Vec::new();

    // Group cases by repo
    let mut cases_by_repo: BTreeMap<String, Vec<&SweBenchCase>> = BTreeMap::new();
    for case in &cases {
        cases_by_repo
            .entry(case.repo.clone())
            .or_default()
            .push(case);
    }

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
            println!("Loading existing Astra index for {}...", repo_name);
            let mut g = astra::storage::load_graph(&repo_config)?;
            g.rebuild_after_deserialize();
            let mut v = astra::storage::load_vector_store(&repo_config)?;
            v.rebuild_index();
            (g, v)
        } else {
            println!("Indexing {} for Astra...", repo_name);
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

        let repo_workspace = Arc::new(repo_workspace);
        let llmg = llmg.clone();

        let semaphore = Arc::new(tokio::sync::Semaphore::new(4));
        let mut handles = Vec::new();

        for case in repo_cases {
            let case = case.clone();
            let astra_engine = astra_engine.clone();
            let rag_index = rag_index.clone();
            let repo_workspace = repo_workspace.clone();
            let llmg = llmg.clone();
            let top_k = args.top_k;
            let permit = semaphore.clone().acquire_owned().await.unwrap();

            handles.push(tokio::spawn(async move {
                let _permit = permit;
                let mut local_records = Vec::new();
                for method in [
                    Method::Astra,
                    Method::Grep,
                    Method::Ripgrep,
                    Method::TraditionalRag,
                ] {
                    let start_total = Instant::now();
                    let retrieved = match method {
                        Method::Astra => run_astra(&astra_engine, &case.query, top_k),
                        Method::Grep => run_grep(&repo_workspace, &case.query, top_k),
                        Method::Ripgrep => run_ripgrep(&repo_workspace, &case.query, top_k),
                        Method::TraditionalRag => Ok(rag_index.search(&case.query, top_k)),
                    }?;

                    let (prompt_tokens, completion_tokens, total_tokens) =
                        if let Some(evaluator) = &llmg {
                            evaluator.evaluate(&case.query, &retrieved.context).await?
                        } else {
                            let prompt =
                                estimate_tokens(&case.query) + estimate_tokens(&retrieved.context);
                            (prompt, 0, prompt)
                        };

                    let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;
                    let recall = compute_recall(&case.oracle_files, &retrieved.files);
                    let oracle_hit = recall > 0.0;
                    local_records.push(BenchmarkRecord {
                        case_id: case.id.clone(),
                        method,
                        retrieval_ms: retrieved.retrieval_ms,
                        total_ms,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        skeleton_tokens: retrieved.skeleton_tokens,
                        body_tokens: retrieved.body_tokens,
                        oracle_hit,
                        recall,
                        retrieved_files: retrieved.files.clone(),
                    });
                }
                Ok::<Vec<BenchmarkRecord>, anyhow::Error>(local_records)
            }));
        }

        for handle in handles {
            if let Ok(Ok(mut local_records)) = handle.await {
                records.append(&mut local_records);
            }
        }
    }

    let output = BenchmarkOutput {
        metadata: BenchmarkMetadata {
            workspace: args.workspace.display().to_string(),
            swe_bench_jsonl: args.swe_bench_jsonl.display().to_string(),
            timestamp_unix_secs: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            cases_evaluated: cases.len(),
            top_k: args.top_k,
            llmg_model: args.llmg_model,
            methodology: "SWE-bench issue retrieval benchmark. Measures retrieval latency and end-to-end (retrieval + optional LLMG completion) latency, reports token usage from LLMG when available, and computes recall@k from patch-only oracle files using path-component matching.".to_string(),
        },
        summary: summarize(&records),
        records,
    };

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, serde_json::to_string_pretty(&output)?).with_context(|| {
        format!(
            "failed to write benchmark output to {}",
            args.output.display()
        )
    })?;
    println!("Benchmark report written to {}", args.output.display());
    Ok(())
}

/// Per-case context token budget.
///
/// Prevents a single outlier case (e.g. a 5 000-line matplotlib class) from
/// blowing up the total token count.  Files are still collected for recall
/// measurement even after the budget is exhausted.
const CONTEXT_TOKEN_BUDGET: u64 = 16_384;

fn run_astra(engine: &SearchEngine, query: &str, top_k: usize) -> Result<RetrievalResult> {
    let start = Instant::now();
    let paths = engine.search(query, top_k)?;
    let mut seen = HashSet::new();
    let mut seen_symbols = HashSet::new();
    let mut files = Vec::new();
    let mut context = String::new();
    let mut context_body_tokens: u64 = 0;
    let mut context_skeleton_tokens: u64 = 0;
    for path in paths {
        if path.nodes.is_empty() {
            continue;
        }
        let mut final_nodes = Vec::new();

        if !path.nodes.is_empty() {
            // 1. Always include the entry point (the symptom)
            final_nodes.push(path.nodes.first().unwrap());

            // 2. If it's a multi-hop path, grab the single most relevant intermediate node
            if path.nodes.len() > 2 {
                let middle_nodes = &path.nodes[1..path.nodes.len() - 1];
                if let Some(best_mid) = middle_nodes
                    .iter()
                    .max_by(|a, b| a.relevance.partial_cmp(&b.relevance).unwrap())
                {
                    final_nodes.push(best_mid);
                }
            }

            // 3. Always include the target node (the root cause)
            if path.nodes.len() > 1 {
                final_nodes.push(path.nodes.last().unwrap());
            }
        }

        let last_idx = final_nodes.len().saturating_sub(1);
        for (idx, node) in final_nodes.into_iter().enumerate() {
            // Always track files for recall measurement regardless of budget
            let normalized = normalize_match_path(&node.file_path);
            if seen.insert(normalized.clone()) {
                files.push(normalized);
            }
            if !seen_symbols.insert(node.symbol_id.clone()) {
                continue;
            }
            // Stop adding context once the token budget is exhausted
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
    Ok(RetrievalResult {
        context,
        files,
        retrieval_ms: start.elapsed().as_secs_f64() * 1000.0,
        skeleton_tokens: context_skeleton_tokens,
        body_tokens: context_body_tokens,
    })
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

fn summarize(records: &[BenchmarkRecord]) -> BTreeMap<Method, MethodSummary> {
    let mut grouped: BTreeMap<Method, Vec<&BenchmarkRecord>> = BTreeMap::new();
    for record in records {
        grouped.entry(record.method).or_default().push(record);
    }

    let mut summary = BTreeMap::new();
    for (method, group) in grouped {
        let mut retrieval_values: Vec<f64> = group.iter().map(|r| r.retrieval_ms).collect();
        let mut total_values: Vec<f64> = group.iter().map(|r| r.total_ms).collect();
        retrieval_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        total_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let runs = group.len();
        let prompt_tokens: u64 = group.iter().map(|r| r.prompt_tokens).sum();
        let completion_tokens: u64 = group.iter().map(|r| r.completion_tokens).sum();
        let total_tokens: u64 = group.iter().map(|r| r.total_tokens).sum();
        let total_skeleton_tokens: u64 = group.iter().map(|r| r.skeleton_tokens).sum();
        let total_body_tokens: u64 = group.iter().map(|r| r.body_tokens).sum();
        let hits = group.iter().filter(|r| r.oracle_hit).count() as f64;
        let recall_sum: f64 = group.iter().map(|r| r.recall).sum();

        summary.insert(
            method,
            MethodSummary {
                runs,
                retrieval_avg_ms: mean(&retrieval_values),
                retrieval_median_ms: percentile(&retrieval_values, 50.0),
                retrieval_p95_ms: percentile(&retrieval_values, 95.0),
                total_avg_ms: mean(&total_values),
                total_p95_ms: percentile(&total_values, 95.0),
                prompt_tokens,
                completion_tokens,
                total_tokens,
                total_skeleton_tokens,
                total_body_tokens,
                oracle_hit_rate: if runs == 0 { 0.0 } else { hits / runs as f64 },
                recall_at_k: if runs == 0 {
                    0.0
                } else {
                    recall_sum / runs as f64
                },
            },
        );
    }
    summary
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let rank = ((percentile / 100.0) * (sorted_values.len().saturating_sub(1)) as f64).round();
    let index = rank.clamp(0.0, (sorted_values.len() - 1) as f64) as usize;
    sorted_values[index]
}

/// Compute recall@k: fraction of oracle files found in retrieved files.
/// Uses proper path-component suffix matching (splits on `/`) to avoid
/// false positives like `query.py` matching `other_query.py`.
fn compute_recall(oracle_files: &HashSet<String>, retrieved_files: &[String]) -> f64 {
    if oracle_files.is_empty() {
        return 0.0;
    }
    let normalized_oracle: Vec<String> = oracle_files
        .iter()
        .map(|file| normalize_match_path(file))
        .collect();
    let normalized_retrieved: Vec<String> = retrieved_files
        .iter()
        .map(|file| normalize_match_path(file))
        .collect();

    let mut hits = 0usize;
    for oracle in &normalized_oracle {
        let oracle_components: Vec<&str> = oracle.split('/').collect();
        for retrieved in &normalized_retrieved {
            let ret_components: Vec<&str> = retrieved.split('/').collect();
            // Match if one path is a suffix of the other at the component level
            if paths_match_by_components(&oracle_components, &ret_components) {
                hits += 1;
                break;
            }
        }
    }
    hits as f64 / normalized_oracle.len() as f64
}

/// Check if two path component lists match: either one is a suffix of the other,
/// or they are equal. This prevents `query.py` from matching `other_query.py`.
fn paths_match_by_components(a: &[&str], b: &[&str]) -> bool {
    if a == b {
        return true;
    }
    let min_len = a.len().min(b.len());
    if min_len == 0 {
        return false;
    }
    // Check if the shorter path is a suffix of the longer
    let a_tail = &a[a.len() - min_len..];
    let b_tail = &b[b.len() - min_len..];
    a_tail == b_tail
}

/// Backward-compatible wrapper: returns true if recall > 0.
#[cfg(test)]
fn has_oracle_hit(oracle_files: &HashSet<String>, retrieved_files: &[String]) -> bool {
    compute_recall(oracle_files, retrieved_files) > 0.0
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

/// Extract code identifiers from the query for use as grep/ripgrep needles.
/// Prioritizes identifiers that look like code (PascalCase, snake_case, dotted
/// paths, quoted strings) over generic English words, since that's what a human
/// would actually grep for.
fn build_text_needles(query: &str) -> Vec<String> {
    let sanitized = sanitize_text_query(query);
    if sanitized.is_empty() {
        return Vec::new();
    }

    let mut needles: Vec<String> = Vec::new();
    let mut seen = HashSet::new();

    // Phase 1: Extract code identifiers (highest priority)
    // -- dotted paths like `django.db.models.query` or `self.rhs.clear_select_clause()`
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

    // -- PascalCase/camelCase identifiers like `ForeignKey`, `QuerySet`, `has_select_fields`
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

    // -- Backtick-quoted code like `process_rhs`
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

    // Phase 2: Fallback — split remaining tokens (only if we haven't found enough)
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

/// Try to add a needle to the list, deduplicating by lowercase.
fn try_add_needle(needles: &mut Vec<String>, seen: &mut HashSet<String>, s: String) {
    let lower = s.to_ascii_lowercase();
    if lower.len() >= 3 && seen.insert(lower) {
        needles.push(s);
    }
}

/// Returns true for common English words that are useless as grep needles.
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
    // Only use "patch" for oracle files, NOT "test_patch".
    // Test files are in obvious locations and inflate hit rates.
    let mut oracle_files = HashSet::new();
    if let Some(patch) = value.get("patch").and_then(Value::as_str) {
        oracle_files.extend(extract_files_from_patch(patch));
    }
    Some(SweBenchCase {
        id,
        repo,
        query,
        oracle_files,
    })
}

fn extract_files_from_patch(patch: &str) -> HashSet<String> {
    let mut files = HashSet::new();
    for line in patch.lines() {
        // Only use +++ b/ lines (the "after" path). --- a/ can be the old path
        // in renames or /dev/null for new files.
        if let Some(file) = line.strip_prefix("+++ b/") {
            let normalized = normalize_match_path(file);
            if normalized != "dev/null" && !normalized.is_empty() {
                files.insert(normalized);
            }
        }
    }
    files
}

/// Approximate BPE token count.
///
/// Code-heavy content (Python, Rust, etc.) averages ~3–3.5 characters per
/// token under common BPE tokenizers (cl100k_base, etc.), while plain English
/// is closer to ~4.  Since benchmark contexts are predominantly source code,
/// we use `len / 3` which slightly over-estimates for English but is accurate
/// for the code-heavy payloads ASTra produces.
fn estimate_tokens(text: &str) -> u64 {
    text.len() as u64 / 3
}

fn parse_args() -> Result<Args> {
    let mut workspace: Option<PathBuf> = None;
    let mut swe_bench_jsonl: Option<PathBuf> = None;
    let mut output = PathBuf::from("benchmarks/reports/benchmark_results.json");
    let mut max_cases = 50usize;
    let mut top_k = 10usize;
    let mut llmg_model: Option<String> = None;
    let mut llmg_max_tokens = 256u32;

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

    Ok(Args {
        workspace,
        swe_bench_jsonl,
        output,
        max_cases,
        top_k,
        llmg_model,
        llmg_max_tokens,
    })
}

fn print_help() {
    println!(
        "Usage:\n  cargo run --bin swe_bench_benchmark -- --workspace <repo> --swe-bench-jsonl <file> [options]\n\nOptions:\n  --output <file>            Output JSON report path (default: benchmarks/reports/benchmark_results.json)\n  --max-cases <n>            Number of SWE-bench cases to evaluate (default: 50)\n  --top-k <n>                Retrieval depth per method (default: 10)\n  --llmg-model <provider/model> Enable LLMG completion scoring and real usage token reporting\n  --llmg-max-tokens <n>      Max completion tokens for LLMG call (default: 256)\n"
    );
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use serde_json::Value;

    use super::{
        build_text_needles, case_from_value, extract_files_from_patch, has_oracle_hit,
        normalize_match_path, percentile, sanitize_text_query,
    };

    #[test]
    fn test_extract_files_from_patch() {
        let patch = "\
diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,3 +1,3 @@
";
        let files = extract_files_from_patch(patch);
        assert!(files.contains("src/main.rs"));
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 95.0), 5.0);
        assert_eq!(percentile(&values, 50.0), 3.0);
    }

    #[test]
    fn test_case_from_value() {
        let value: Value = serde_json::json!({
            "instance_id": "abc",
            "repo": "test/repo",
            "problem_statement": "fix parser panic",
            "patch": "+++ b/src/lib.rs\n"
        });
        let case = case_from_value(&value).unwrap();
        assert_eq!(case.id, "abc");
        assert_eq!(case.query, "fix parser panic");
        assert!(case.oracle_files.contains("src/lib.rs"));
    }

    #[test]
    fn test_has_oracle_hit_normalizes_windows_paths() {
        let oracle = HashSet::from(["src/module/file.py".to_string()]);
        let retrieved = vec![".\\src\\module\\file.py".to_string()];
        assert!(has_oracle_hit(&oracle, &retrieved));
    }

    #[test]
    fn test_normalize_match_path() {
        assert_eq!(normalize_match_path("./src/main.rs"), "src/main.rs");
        assert_eq!(normalize_match_path(".\\src\\main.rs"), "src/main.rs");
        assert_eq!(normalize_match_path("/src/main.rs"), "src/main.rs");
    }

    #[test]
    fn test_sanitize_text_query() {
        assert_eq!(
            sanitize_text_query("line one\nline\ttwo"),
            "line one line two"
        );
    }

    #[test]
    fn test_has_oracle_hit_bidirectional() {
        // Retrieved file is a suffix of oracle path
        let oracle = HashSet::from(["django/db/models/query.py".to_string()]);
        let retrieved = vec!["db/models/query.py".to_string()];
        assert!(has_oracle_hit(&oracle, &retrieved));

        // Retrieved file is a prefix of oracle path (original direction)
        let oracle2 = HashSet::from(["models/query.py".to_string()]);
        let retrieved2 = vec!["django/db/models/query.py".to_string()];
        assert!(has_oracle_hit(&oracle2, &retrieved2));
    }

    #[test]
    fn test_build_text_needles_extracts_code_identifiers() {
        let query = "The ForeignKey lookup fails when using isnull with QuerySet";
        let needles = build_text_needles(query);
        assert!(!needles.is_empty());
        // Should extract PascalCase identifiers
        let lower_needles: Vec<String> = needles.iter().map(|n| n.to_ascii_lowercase()).collect();
        assert!(
            lower_needles.iter().any(|n| n.contains("foreignkey")),
            "should find ForeignKey, got: {:?}",
            needles
        );
        assert!(
            lower_needles.iter().any(|n| n.contains("queryset")),
            "should find QuerySet, got: {:?}",
            needles
        );
    }

    // ── New tests for fixed functions ──

    #[test]
    fn test_compute_recall_full_match() {
        use super::compute_recall;
        let oracle = HashSet::from(["src/main.rs".to_string(), "src/lib.rs".to_string()]);
        let retrieved = vec![
            "src/main.rs".to_string(),
            "src/lib.rs".to_string(),
            "src/other.rs".to_string(),
        ];
        assert_eq!(compute_recall(&oracle, &retrieved), 1.0);
    }

    #[test]
    fn test_compute_recall_partial_match() {
        use super::compute_recall;
        let oracle = HashSet::from(["src/main.rs".to_string(), "src/lib.rs".to_string()]);
        let retrieved = vec!["src/main.rs".to_string()];
        assert_eq!(compute_recall(&oracle, &retrieved), 0.5);
    }

    #[test]
    fn test_compute_recall_no_match() {
        use super::compute_recall;
        let oracle = HashSet::from(["src/main.rs".to_string()]);
        let retrieved = vec!["src/other.rs".to_string()];
        assert_eq!(compute_recall(&oracle, &retrieved), 0.0);
    }

    #[test]
    fn test_paths_match_by_components_prevents_false_positives() {
        use super::paths_match_by_components;
        // "query.py" should NOT match "other_query.py"
        let a: Vec<&str> = "query.py".split('/').collect();
        let b: Vec<&str> = "other_query.py".split('/').collect();
        assert!(!paths_match_by_components(&a, &b));

        // "query.py" SHOULD match "db/models/query.py" (component-level suffix)
        let a2: Vec<&str> = "query.py".split('/').collect();
        let b2: Vec<&str> = "db/models/query.py".split('/').collect();
        assert!(paths_match_by_components(&a2, &b2));
    }

    #[test]
    fn test_extract_files_from_patch_ignores_dev_null() {
        let patch = "\
diff --git /dev/null b/src/new_file.rs
--- /dev/null
+++ b/src/new_file.rs
@@ -0,0 +1,5 @@
";
        let files = extract_files_from_patch(patch);
        assert!(files.contains("src/new_file.rs"));
        assert!(!files.contains("dev/null"));
    }

    #[test]
    fn test_extract_files_from_patch_ignores_old_path() {
        let patch = "\
diff --git a/src/old.rs b/src/new.rs
--- a/src/old.rs
+++ b/src/new.rs
@@ -1,3 +1,3 @@
";
        let files = extract_files_from_patch(patch);
        assert!(files.contains("src/new.rs"));
        // --- a/ lines should no longer be captured
        assert!(!files.contains("src/old.rs"));
    }

    #[test]
    fn test_case_from_value_excludes_test_patch() {
        let value: Value = serde_json::json!({
            "instance_id": "test_case",
            "repo": "test/repo",
            "problem_statement": "some bug",
            "patch": "+++ b/src/fix.py\n",
            "test_patch": "+++ b/tests/test_fix.py\n"
        });
        let case = case_from_value(&value).unwrap();
        assert!(case.oracle_files.contains("src/fix.py"));
        assert!(
            !case.oracle_files.contains("tests/test_fix.py"),
            "test_patch files should not be in oracle set"
        );
    }

    #[test]
    fn test_estimate_tokens_bpe_approximation() {
        use super::estimate_tokens;
        // "hello world" = 11 chars → 11/3 = 3 tokens
        assert_eq!(estimate_tokens("hello world"), 3);
        // Empty string → 0/3 = 0
        assert_eq!(estimate_tokens(""), 0);
        // 100 chars → 100/3 = 33 tokens
        let long_text = "a".repeat(100);
        assert_eq!(estimate_tokens(&long_text), 33);
    }
}
