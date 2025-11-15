use std::sync::Arc;

use tokio::task;

use crate::{
    config::AppConfig,
    error::ServiceError,
    model::{
        GenerationRequest, GenerationResponse, ModelMetadata, loader::ModelArtifacts,
        loader::ModelInstance,
    },
};

pub struct ModelRegistry {
    artifacts: Arc<ModelArtifacts>,
}

impl ModelRegistry {
    pub fn initialize(config: &AppConfig) -> Result<Self, ServiceError> {
        let artifacts = ModelArtifacts::load(config)?;
        Ok(Self {
            artifacts: Arc::new(artifacts),
        })
    }

    pub fn metadata(&self) -> (Option<ModelMetadata>, Option<ModelMetadata>) {
        let quantized = self.artifacts.quantized.as_ref().map(|m| m.metadata());
        let baseline = self
            .artifacts
            .baseline
            .as_ref()
            .map(|model| model.metadata());
        (quantized, baseline)
    }

    pub fn has_baseline(&self) -> bool {
        self.artifacts.baseline.is_some()
    }

    pub fn has_quantized(&self) -> bool {
        self.artifacts.quantized.is_some()
    }

    pub async fn generate_quantized(
        &self,
        request: GenerationRequest,
        config: &AppConfig,
    ) -> Result<GenerationResponse, ServiceError> {
        let model = self
            .artifacts
            .quantized
            .clone()
            .ok_or_else(|| {
                ServiceError::Other("Quantized model not available".to_string())
            })?;
        self.spawn_inference(model, request, config).await
    }

    pub async fn generate_baseline(
        &self,
        request: GenerationRequest,
        config: &AppConfig,
    ) -> Result<GenerationResponse, ServiceError> {
        let model = self
            .artifacts
            .baseline
            .clone()
            .ok_or_else(|| ServiceError::ModelLoading)?;
        self.spawn_inference(model, request, config).await
    }

    async fn spawn_inference(
        &self,
        model: Arc<ModelInstance>,
        request: GenerationRequest,
        config: &AppConfig,
    ) -> Result<GenerationResponse, ServiceError> {
        let tokenizer = self.artifacts.tokenizer.clone();
        let prompt = request.prompt;
        let max_new_tokens = request.max_new_tokens.unwrap_or(config.max_new_tokens);
        let temperature = request.temperature.unwrap_or(config.temperature);
        let top_k = request.top_k.unwrap_or(config.top_k);

        task::spawn_blocking(move || {
            model.generate(&tokenizer, &prompt, max_new_tokens, temperature, top_k)
        })
        .await
        .map_err(|err| ServiceError::Inference(format!("inference task failed: {err}")))?
    }
}
