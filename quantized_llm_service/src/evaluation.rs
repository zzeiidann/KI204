use std::{fs, path::Path, sync::Arc};

use serde::Serialize;

use crate::{
    config::AppConfig,
    error::ServiceError,
    model::{GenerationRequest, GenerationResponse, ModelRegistry},
};

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkSample {
    pub prompt: String,
    pub reference_substring: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleReport {
    pub prompt: String,
    pub quantized: GenerationResponse,
    pub baseline: Option<GenerationResponse>,
    pub reference_match_quantized: Option<bool>,
    pub reference_match_baseline: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateMetrics {
    pub quantized_avg_latency_ms: f64,
    pub quantized_avg_tokens_per_s: f64,
    pub baseline_avg_latency_ms: Option<f64>,
    pub baseline_avg_tokens_per_s: Option<f64>,
    pub quantized_reference_match_rate: Option<f64>,
    pub baseline_reference_match_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationReport {
    pub samples: Vec<SampleReport>,
    pub aggregate: AggregateMetrics,
}

pub async fn run_benchmark(
    registry: Arc<ModelRegistry>,
    config: &AppConfig,
    samples: Vec<BenchmarkSample>,
) -> Result<EvaluationReport, ServiceError> {
    if samples.is_empty() {
        return Err(ServiceError::BadRequest(
            "at least one benchmark sample is required".into(),
        ));
    }

    let mut reports = Vec::with_capacity(samples.len());

    for sample in samples {
        let request = GenerationRequest {
            prompt: sample.prompt.clone(),
            max_new_tokens: Some(config.max_new_tokens),
            temperature: Some(config.temperature),
            top_k: Some(config.top_k),
        };

        let quantized = registry.generate_quantized(request, config).await?;

        let baseline = if registry.has_baseline() {
            let request = GenerationRequest {
                prompt: sample.prompt.clone(),
                max_new_tokens: Some(config.max_new_tokens),
                temperature: Some(config.temperature),
                top_k: Some(config.top_k),
            };
            Some(registry.generate_baseline(request, config).await?)
        } else {
            None
        };

        let reference_match_quantized = sample.reference_substring.as_ref().map(|needle| {
            quantized
                .completion
                .to_lowercase()
                .contains(&needle.to_lowercase())
        });
        let reference_match_baseline = sample.reference_substring.as_ref().and_then(|needle| {
            baseline.as_ref().map(|resp| {
                resp.completion
                    .to_lowercase()
                    .contains(&needle.to_lowercase())
            })
        });

        reports.push(SampleReport {
            prompt: sample.prompt,
            quantized,
            baseline,
            reference_match_quantized,
            reference_match_baseline,
        });
    }

    let aggregate = summarize(&reports);

    Ok(EvaluationReport {
        samples: reports,
        aggregate,
    })
}

pub fn load_samples_from_path(path: &Path) -> Result<Vec<BenchmarkSample>, ServiceError> {
    let raw = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| ServiceError::BadRequest(format!("invalid benchmark file: {e}")))?;

    match value {
        serde_json::Value::Array(items) => {
            let mut samples = Vec::with_capacity(items.len());
            for (idx, item) in items.into_iter().enumerate() {
                let prompt = item.get("prompt").and_then(|v| v.as_str()).ok_or_else(|| {
                    ServiceError::BadRequest(format!(
                        "benchmark item {idx} missing string field 'prompt'"
                    ))
                })?;
                let reference_substring = item
                    .get("reference_substring")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                samples.push(BenchmarkSample {
                    prompt: prompt.to_string(),
                    reference_substring,
                });
            }
            Ok(samples)
        }
        _ => Err(ServiceError::BadRequest(
            "benchmark file must be a JSON array".into(),
        )),
    }
}

pub fn fallback_samples() -> Vec<BenchmarkSample> {
    vec![
        BenchmarkSample {
            prompt: "Explain the benefits of quantizing a transformer model to int8 precision."
                .to_string(),
            reference_substring: Some("quant".to_string()),
        },
        BenchmarkSample {
            prompt: "Summarize the rust borrow checker in one sentence.".to_string(),
            reference_substring: Some("borrow".to_string()),
        },
        BenchmarkSample {
            prompt: "Write a haiku about efficient machine learning inference.".to_string(),
            reference_substring: Some("haiku".to_string()),
        },
    ]
}

fn summarize(reports: &[SampleReport]) -> AggregateMetrics {
    let quantized_avg_latency_ms = mean(reports.iter().map(|r| r.quantized.total_time_ms as f64));
    let quantized_avg_tokens_per_s = mean(reports.iter().map(|r| r.quantized.tokens_per_second));

    let baseline_latencies: Vec<f64> = reports
        .iter()
        .filter_map(|r| r.baseline.as_ref())
        .map(|r| r.total_time_ms as f64)
        .collect();
    let baseline_avg_latency_ms = if baseline_latencies.is_empty() {
        None
    } else {
        Some(mean(baseline_latencies.into_iter()))
    };

    let baseline_tps: Vec<f64> = reports
        .iter()
        .filter_map(|r| r.baseline.as_ref())
        .map(|r| r.tokens_per_second)
        .collect();
    let baseline_avg_tokens_per_s = if baseline_tps.is_empty() {
        None
    } else {
        Some(mean(baseline_tps.into_iter()))
    };

    let quantized_reference_match_rate =
        compute_match_rate(reports.iter().filter_map(|r| r.reference_match_quantized));
    let baseline_reference_match_rate =
        compute_match_rate(reports.iter().filter_map(|r| r.reference_match_baseline));

    AggregateMetrics {
        quantized_avg_latency_ms,
        quantized_avg_tokens_per_s,
        baseline_avg_latency_ms,
        baseline_avg_tokens_per_s,
        quantized_reference_match_rate,
        baseline_reference_match_rate,
    }
}

fn mean<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut count = 0usize;
    let mut acc = 0.0;
    for value in values {
        count += 1;
        acc += value;
    }
    if count == 0 { 0.0 } else { acc / count as f64 }
}

fn compute_match_rate<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = bool>,
{
    let mut count = 0usize;
    let mut matches = 0usize;
    for value in values {
        count += 1;
        if value {
            matches += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(matches as f64 / count as f64)
    }
}
