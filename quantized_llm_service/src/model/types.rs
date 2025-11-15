use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerationResponse {
    pub prompt: String,
    pub completion: String,
    pub tokens_generated: usize,
    pub total_time_ms: u128,
    pub tokens_per_second: f64,
    pub model: ModelMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelMetadata {
    pub name: String,
    pub quantized: bool,
    pub dtype: String,
    pub size_bytes: u64,
}
