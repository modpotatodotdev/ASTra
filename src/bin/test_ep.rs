use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use ort::execution_providers::CPUExecutionProvider;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

fn main() {
    tracing_subscriber::fmt::init();
    println!("Starting fastembed CUDA test...");
    #[cfg(feature = "cuda")]
    let execution_providers = vec![
        CUDAExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
    ];
    #[cfg(not(feature = "cuda"))]
    let execution_providers = vec![CPUExecutionProvider::default().build()];

    let options = InitOptions::new(EmbeddingModel::BGEBaseENV15)
        .with_show_download_progress(true)
        .with_execution_providers(execution_providers);
    let model = TextEmbedding::try_new(options);

    match model {
        Ok(m) => {
            println!("Model initialized! Running a test embed...");
            let result = m.embed(vec!["hello world test"], None);
            match result {
                Ok(embeddings) => println!("Embedding succeeded! dim={}", embeddings[0].len()),
                Err(e) => println!("Embedding failed: {:?}", e),
            }
        }
        Err(e) => println!("Init failed: {:?}", e),
    }
}
