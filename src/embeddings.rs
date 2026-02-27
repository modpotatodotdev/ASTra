/// Trait for generating embeddings from text.
pub trait Embedder: Send + Sync {
    /// Generate an embedding vector for the given text.
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts.
    fn embed_batch(&self, texts: Vec<&str>) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embedding dimensionality.
    fn dim(&self) -> usize;
}

pub const DEFAULT_LOCAL_DIM: usize = 768;

/// Instantiate the appropriate embedder based on the provided configuration setting
/// and the active feature flags at compile time.
pub fn build_embedder(provider: &str) -> anyhow::Result<Box<dyn Embedder>> {
    match provider {
        #[cfg(feature = "local")]
        "local" => {
            log::info!("Loading local semantic embedder (BAAI/bge-base)...");
            Ok(Box::new(SemanticEmbedder::try_new()?))
        }
        #[cfg(feature = "openrouter")]
        "openrouter" => {
            log::info!("Loading OpenRouter API embedder...");
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY environment variable required for 'openrouter' provider"))?;
            let model = std::env::var("OPENROUTER_MODEL")
                .unwrap_or_else(|_| "openai/text-embedding-3-small".to_string());
            Ok(Box::new(OpenRouterEmbedder::try_new(api_key, model)?))
        }
        other => anyhow::bail!(
            "Configured embedding provider '{}' is unsupported or its feature flag was omitted during compilation.",
            other
        ),
    }
}

#[cfg(feature = "local")]
/// Semantic embedder backed by `BAAI/bge-base-en-v1.5` via ONNX Runtime (fastembed).
///
/// Produces 768-dimensional dense embeddings with true semantic understanding,
/// enabling multi-hop XY-problem resolution where the query shares no keywords
/// with the target file. The model is downloaded from HuggingFace Hub on first
/// use and cached locally in `.fastembed_cache/`.
pub struct SemanticEmbedder {
    inner: fastembed::TextEmbedding,
}

#[cfg(feature = "local")]
impl SemanticEmbedder {
    /// Initialise the embedder, downloading the model if not already cached.
    ///
    /// Returns an error if the model cannot be loaded or downloaded.
    pub fn try_new() -> anyhow::Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        use ort::execution_providers::CPUExecutionProvider;

        #[cfg(feature = "cuda")]
        let execution_providers = {
            use ort::execution_providers::CUDAExecutionProvider;
            vec![
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ]
        };
        #[cfg(not(feature = "cuda"))]
        let execution_providers = vec![CPUExecutionProvider::default().build()];

        let init_options = InitOptions::new(EmbeddingModel::BGEBaseENV15)
            .with_show_download_progress(true)
            .with_execution_providers(execution_providers);

        let inner = TextEmbedding::try_new(init_options)
            .map_err(|e| anyhow::anyhow!("Failed to load BGEBaseENV15: {}", e))?;
        Ok(Self { inner })
    }
}

#[cfg(feature = "local")]
impl Embedder for SemanticEmbedder {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.inner
            .embed(vec![text], None)
            .map(|mut vecs| vecs.remove(0))
            .map_err(|e| anyhow::anyhow!("SemanticEmbedder inference failed: {}", e))
    }

    fn embed_batch(&self, texts: Vec<&str>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.inner
            .embed(texts, None)
            .map_err(|e| anyhow::anyhow!("SemanticEmbedder batch inference failed: {}", e))
    }

    fn dim(&self) -> usize {
        DEFAULT_LOCAL_DIM
    }
}

#[cfg(feature = "openrouter")]
pub struct OpenRouterEmbedder {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
    dim: usize,
}

#[cfg(feature = "openrouter")]
#[derive(serde::Deserialize)]
struct OpenRouterModel {
    id: String,
    name: String,
    #[serde(default)]
    architecture: Option<OpenRouterArchitecture>,
}

#[cfg(feature = "openrouter")]
#[derive(serde::Deserialize)]
struct OpenRouterArchitecture {
    #[serde(default)]
    output_modalities: Vec<String>,
}

#[cfg(feature = "openrouter")]
#[derive(serde::Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[cfg(feature = "openrouter")]
#[derive(serde::Deserialize)]
struct OpenRouterEmbeddingResponse {
    data: Vec<OpenRouterEmbeddingData>,
}

#[cfg(feature = "openrouter")]
#[derive(serde::Deserialize)]
struct OpenRouterEmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(feature = "openrouter")]
impl OpenRouterEmbedder {
    /// Fetch available embedding models from OpenRouter and validate the requested model.
    ///
    /// Returns model info if found and confirmed to support embeddings.
    fn validate_embedding_model(
        client: &reqwest::blocking::Client,
        api_key: &str,
        model_id: &str,
    ) -> anyhow::Result<OpenRouterModel> {
        let res = client
            .get("https://openrouter.ai/api/v1/embeddings/models")
            .header("Authorization", format!("Bearer {}", api_key))
            .send()?;

        if !res.status().is_success() {
            let status = res.status();
            let err_text = res.text().unwrap_or_default();
            anyhow::bail!("OpenRouter models list error ({}): {}", status, err_text);
        }

        let parsed: OpenRouterModelsResponse = res.json()?;
        let model = parsed
            .data
            .into_iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Model '{}' not found in OpenRouter embeddings models. \
                     Use the OPENROUTER_MODEL env var to specify a valid embedding model.",
                    model_id
                )
            })?;

        let supports_embeddings = model
            .architecture
            .as_ref()
            .map(|arch| arch.output_modalities.iter().any(|m| m == "embeddings"))
            .unwrap_or(false);

        if !supports_embeddings {
            anyhow::bail!(
                "Model '{}' ({}) does not support embeddings output modality. \
                 Choose a model with 'embeddings' output modality.",
                model_id,
                model.name
            );
        }

        log::info!(
            "Validated OpenRouter embedding model: {} ({})",
            model_id,
            model.name
        );
        Ok(model)
    }

    /// Validate the model, then ping the API to determine embedding dimensionality.
    pub fn try_new(api_key: String, model: String) -> anyhow::Result<Self> {
        use serde_json::json;

        let client = reqwest::blocking::Client::new();

        Self::validate_embedding_model(&client, &api_key, &model)?;

        let res = client
            .post("https://openrouter.ai/api/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&json!({
                "model": model,
                "input": ["ping"]
            }))
            .send()?;

        if !res.status().is_success() {
            let status = res.status();
            let err_text = res.text().unwrap_or_default();
            anyhow::bail!("OpenRouter API error on setup ({}): {}", status, err_text);
        }

        let parsed: OpenRouterEmbeddingResponse = res.json()?;
        let dim = parsed
            .data
            .first()
            .ok_or_else(|| anyhow::anyhow!("Empty embedding data returned during ping"))?
            .embedding
            .len();

        log::info!(
            "OpenRouter embedder initialized: model={}, dim={}",
            model,
            dim
        );

        Ok(Self {
            client,
            api_key,
            model,
            dim,
        })
    }

    fn do_request(&self, input: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        use serde_json::json;
        let res = self
            .client
            .post("https://openrouter.ai/api/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&json!({
                "model": self.model,
                "input": input
            }))
            .send()?;

        if !res.status().is_success() {
            anyhow::bail!("OpenRouter API returned {}", res.status());
        }

        let parsed: OpenRouterEmbeddingResponse = res.json()?;
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[cfg(feature = "openrouter")]
impl Embedder for OpenRouterEmbedder {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.do_request(&[text]).map(|mut vecs| vecs.remove(0))
    }

    fn embed_batch(&self, texts: Vec<&str>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.do_request(&texts)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Compute cosine similarity between two f32 vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Cosine similarity between an f32 query and an f16-quantized stored vector.
///
/// The query remains f32 throughout; f16 elements are up-converted on the fly
/// with no intermediate allocation. Accumulation uses f32 for numerical stability.
pub fn cosine_similarity_f16(a: &[f32], b: &[half::f16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let bf = bi.to_f32();
        dot += ai * bf;
        norm_a_sq += ai * ai;
        norm_b_sq += bf * bf;
    }
    let denom = norm_a_sq.sqrt() * norm_b_sq.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(all(test, feature = "local"))]
mod tests {
    use super::{cosine_similarity, Embedder, SemanticEmbedder, DEFAULT_LOCAL_DIM};

    #[test]
    fn test_semantic_embedder_dim() {
        let embedder = SemanticEmbedder::try_new().expect("model should load");
        let vec = embedder
            .embed("fn hello_world() { println!(\"hi\"); }")
            .unwrap();
        assert_eq!(vec.len(), DEFAULT_LOCAL_DIM);
    }

    #[test]
    fn test_similar_code_has_higher_similarity() {
        let embedder = SemanticEmbedder::try_new().expect("model should load");
        let v1 = embedder
            .embed("fn fetch_user(id: u64) -> User { db.query(id) }")
            .unwrap();
        let v2 = embedder
            .embed("fn get_user(user_id: u64) -> User { database.find(user_id) }")
            .unwrap();
        let v3 = embedder
            .embed("fn calculate_tax(amount: f64) -> f64 { amount * 0.2 }")
            .unwrap();

        let sim_12 = cosine_similarity(&v1, &v2);
        let sim_13 = cosine_similarity(&v1, &v3);

        assert!(
            sim_12 > sim_13,
            "similar functions should have higher similarity: {} vs {}",
            sim_12,
            sim_13
        );
    }

    #[test]
    fn test_embedding_is_normalized() {
        let embedder = SemanticEmbedder::try_new().expect("model should load");
        let vec = embedder.embed("some code here").unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "embedding should be L2-normalized"
        );
    }
}
