use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use parking_lot::RwLock;
use serde::Serialize;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::{
    config::AppConfig,
    error::ServiceError,
    evaluation::{EvaluationReport, fallback_samples, load_samples_from_path, run_benchmark},
    model::{GenerationRequest, ModelRegistry},
    quantization::QuantizationSummary,
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub registry: Arc<ModelRegistry>,
    pub evaluation: Arc<RwLock<Option<EvaluationReport>>>,
}

#[derive(Serialize)]
struct MetadataResponse {
    quantized: Option<crate::model::ModelMetadata>,
    baseline: Option<crate::model::ModelMetadata>,
    quantization: Option<QuantizationSummary>,
    evaluation: Option<EvaluationReport>,
}

pub fn build_router(config: Arc<AppConfig>, registry: Arc<ModelRegistry>) -> Router {
    let state = AppState {
        evaluation: Arc::new(RwLock::new(None)),
        registry,
        config,
    };

    Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate_quantized))
        .route("/generate/baseline", post(generate_baseline))
        .route("/metadata", get(metadata))
        .route("/evaluate", post(run_evaluation))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

async fn health() -> &'static str {
    "ok"
}

async fn generate_quantized(
    State(state): State<AppState>,
    Json(request): Json<GenerationRequest>,
) -> Result<Json<crate::model::GenerationResponse>, ServiceError> {
    // Use quantized model if available, otherwise fallback to baseline
    let response = if state.registry.has_quantized() {
        state
            .registry
            .generate_quantized(request, &state.config)
            .await?
    } else {
        state
            .registry
            .generate_baseline(request, &state.config)
            .await?
    };
    Ok(Json(response))
}

async fn generate_baseline(
    State(state): State<AppState>,
    Json(request): Json<GenerationRequest>,
) -> Result<Json<crate::model::GenerationResponse>, ServiceError> {
    if !state.registry.has_baseline() {
        return Err(ServiceError::BadRequest(
            "baseline model not available".into(),
        ));
    }
    let response = state
        .registry
        .generate_baseline(request, &state.config)
        .await?;
    Ok(Json(response))
}

async fn metadata(State(state): State<AppState>) -> Json<MetadataResponse> {
    let (quantized, baseline) = state.registry.metadata();
    let summarised = if let Some(ref q) = quantized {
        Some(QuantizationSummary::from_metadata(q, baseline.as_ref()))
    } else {
        None
    };
    let evaluation = state.evaluation.read().clone();

    Json(MetadataResponse {
        quantized,
        baseline,
        quantization: summarised,
        evaluation,
    })
}

async fn run_evaluation(
    State(state): State<AppState>,
) -> Result<Json<EvaluationReport>, ServiceError> {
    let samples = if let Some(path) = state.config.eval_prompts_path.as_ref() {
        load_samples_from_path(path)?
    } else {
        fallback_samples()
    };

    info!(count = samples.len(), "running evaluation benchmark");

    let report = run_benchmark(state.registry.clone(), &state.config, samples).await?;
    state.evaluation.write().replace(report.clone());

    Ok(Json(report))
}
