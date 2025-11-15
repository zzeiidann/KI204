use serde::Serialize;

use crate::model::ModelMetadata;

#[derive(Debug, Serialize)]
pub struct QuantizationSummary {
    pub baseline_size_bytes: Option<u64>,
    pub quantized_size_bytes: u64,
    pub size_reduction_percent: Option<f64>,
}

impl QuantizationSummary {
    pub fn from_metadata(
        quantized: &ModelMetadata,
        baseline: Option<&ModelMetadata>,
    ) -> QuantizationSummary {
        let baseline_size = baseline.map(|m| m.size_bytes);
        let reduction = baseline_size.map(|baseline| {
            if baseline == 0 {
                0.0
            } else {
                let diff = baseline.saturating_sub(quantized.size_bytes) as f64;
                (diff / baseline as f64) * 100.0
            }
        });

        QuantizationSummary {
            baseline_size_bytes: baseline_size,
            quantized_size_bytes: quantized.size_bytes,
            size_reduction_percent: reduction,
        }
    }
}
