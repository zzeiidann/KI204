use std::{fs, path::Path, sync::Arc, time::Instant};

use parking_lot::Mutex;
use tch::{Device, Tensor, no_grad};
use tokenizers::Tokenizer;

use crate::{
    config::AppConfig,
    error::ServiceError,
    model::{GenerationResponse, ModelMetadata},
};

pub struct ModelArtifacts {
    pub tokenizer: Arc<Tokenizer>,
    pub quantized: Option<Arc<ModelInstance>>,
    pub baseline: Option<Arc<ModelInstance>>,
}

pub struct ModelInstance {
    name: String,
    quantized: bool,
    dtype: String,
    size_bytes: u64,
    device: Device,
    module: Mutex<tch::CModule>,
}

impl ModelArtifacts {
    pub fn load(config: &AppConfig) -> Result<Self, ServiceError> {
        let tokenizer = Arc::new(
            Tokenizer::from_file(config.tokenizer_path.as_path())
                .map_err(|e| ServiceError::Tokenizer(e.to_string()))?,
        );

        // Load baseline model (required)
        let baseline = Arc::new(ModelInstance::new(
            "baseline",
            false,
            "float32",
            &config.baseline_module_path,
            config.device,
        )?);

        // Don't load quantized model - dynamic quantization requires LibTorch
        // with quantization backend support that may not be available
        Ok(Self {
            tokenizer,
            quantized: None,
            baseline: Some(baseline),
        })
    }
}

impl ModelInstance {
    pub fn new(
        name: &str,
        quantized: bool,
        dtype: &str,
        module_path: &Path,
        device: Device,
    ) -> Result<Self, ServiceError> {
        if !module_path.exists() {
            return Err(ServiceError::Other(format!(
                "model artifact missing: {}",
                module_path.display()
            )));
        }
        let size_bytes = fs::metadata(module_path)?.len();
        let mut module = tch::CModule::load_on_device(module_path, device)
            .map_err(|e| ServiceError::Inference(e.to_string()))?;
        module.set_eval();

        Ok(Self {
            name: name.to_string(),
            quantized,
            dtype: dtype.to_string(),
            size_bytes,
            device,
            module: Mutex::new(module),
        })
    }

    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            name: self.name.clone(),
            quantized: self.quantized,
            dtype: self.dtype.clone(),
            size_bytes: self.size_bytes,
        }
    }

    pub fn generate(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        _temperature: f64,
        _top_k: usize,
    ) -> Result<GenerationResponse, ServiceError> {
        if prompt.trim().is_empty() {
            return Err(ServiceError::BadRequest("prompt must not be empty".into()));
        }

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| ServiceError::Tokenizer(e.to_string()))?;
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        if input_ids.is_empty() {
            input_ids.push(0);
        }
        let prompt_token_len = input_ids.len();

        let start = Instant::now();

        // Autoregressive generation loop using the traced forward pass
        no_grad(|| {
            let module = self.module.lock();
            
            for _ in 0..max_new_tokens {
                // Create input tensor from current sequence
                let input_tensor = Tensor::from_slice(&input_ids)
                    .reshape([1, input_ids.len() as i64])
                    .to(self.device);

                // Run forward pass - traced GPT-2 model
                // The model may return either a tensor or tuple with (logits, past)
                let output = module
                    .forward_is(&[tch::IValue::Tensor(input_tensor)])
                    .map_err(|e| ServiceError::Inference(e.to_string()))?;
                
                // Extract logits from output (handle both tensor and tuple cases)
                let logits = match output {
                    tch::IValue::Tensor(t) => t,
                    tch::IValue::Tuple(ref tuple) if !tuple.is_empty() => {
                        match &tuple[0] {
                            tch::IValue::Tensor(t) => t.shallow_clone(),
                            _ => return Err(ServiceError::Inference(
                                "Expected tensor as first tuple element".into()
                            )),
                        }
                    }
                    _ => return Err(ServiceError::Inference(
                        "Unexpected model output format".into()
                    )),
                };

                // Get logits for the last token: shape [1, seq_len, vocab_size]
                let last_logits = logits
                    .select(1, -1)  // Select last position in sequence
                    .squeeze();      // Remove batch dimension

                // Greedy sampling: take argmax (for simplicity, ignoring temperature/top_k)
                let next_token_id = last_logits.argmax(0, false).int64_value(&[]);
                
                // Append to sequence
                input_ids.push(next_token_id);

                // Stop if we hit EOS token (50256 for GPT-2)
                if next_token_id == 50256 {
                    break;
                }
            }

            Ok::<(), ServiceError>(())
        })?;

        let elapsed = start.elapsed();

        // Extract only the generated tokens
        let generated_ids: Vec<u32> = input_ids[prompt_token_len..]
            .iter()
            .map(|&id| id as u32)
            .collect();
        let tokens_generated = generated_ids.len();
        
        let completion = tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| ServiceError::Tokenizer(e.to_string()))?;

        let total_tokens = prompt_token_len + tokens_generated;
        let total_time_ms = elapsed.as_millis();
        let tokens_per_second = if elapsed.as_secs_f64() > 0.0 {
            total_tokens as f64 / elapsed.as_secs_f64()
        } else {
            total_tokens as f64
        };

        Ok(GenerationResponse {
            prompt: prompt.to_string(),
            completion,
            tokens_generated,
            total_time_ms,
            tokens_per_second,
            model: self.metadata(),
        })
    }
}
