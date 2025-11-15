use std::{
    env,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    time::Duration,
};

#[cfg(feature = "tch-backend")]
use tch::Device;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub listen_addr: SocketAddr,
    pub model_id: String,
    pub revision: Option<String>,
    pub baseline_module_path: PathBuf,
    pub quantized_module_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub eval_prompts_path: Option<PathBuf>,
    pub eval_reference_path: Option<PathBuf>,
    pub eval_warmup_iters: usize,
    pub eval_benchmark_iters: usize,
    pub eval_timeout: Duration,
    #[cfg(feature = "tch-backend")]
    pub device: Device,
}

impl AppConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let listen_addr = env::var("SERVER_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8080".into())
            .parse()
            .unwrap_or_else(|_| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080));

        let model_id = env::var("MODEL_ID").unwrap_or_else(|_| "distilgpt2".to_string());
        let revision = env::var("MODEL_REVISION").ok();

        let baseline_module_path = PathBuf::from(
            env::var("BASELINE_MODULE_PATH")
                .unwrap_or_else(|_| "models/distilgpt2_baseline.ts".to_string()),
        );
        let quantized_module_path = PathBuf::from(
            env::var("QUANTIZED_MODULE_PATH")
                .unwrap_or_else(|_| "models/distilgpt2_quantized.ts".to_string()),
        );
        let tokenizer_path = PathBuf::from(
            env::var("TOKENIZER_PATH").unwrap_or_else(|_| "models/tokenizer.json".to_string()),
        );

        let max_new_tokens = env::var("MAX_NEW_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let temperature = env::var("TEMPERATURE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.8);
        let top_k = env::var("TOP_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40);

        let eval_prompts_path = env::var("EVAL_PROMPTS_PATH").ok().map(PathBuf::from);
        let eval_reference_path = env::var("EVAL_REFERENCE_PATH").ok().map(PathBuf::from);
        let eval_warmup_iters = env::var("EVAL_WARMUP_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let eval_benchmark_iters = env::var("EVAL_BENCHMARK_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let eval_timeout = env::var("EVAL_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(30));

        #[cfg(feature = "tch-backend")]
        let device = {
            let raw = env::var("DEVICE").unwrap_or_else(|_| "cpu".into());
            parse_device(&raw)
        };

        Ok(Self {
            listen_addr,
            model_id,
            revision,
            baseline_module_path,
            quantized_module_path,
            tokenizer_path,
            max_new_tokens,
            temperature,
            top_k,
            eval_prompts_path,
            eval_reference_path,
            eval_warmup_iters,
            eval_benchmark_iters,
            eval_timeout,
            #[cfg(feature = "tch-backend")]
            device,
        })
    }
}

#[cfg(feature = "tch-backend")]
fn parse_device(raw: &str) -> Device {
    let lower = raw.to_lowercase();
    if lower == "cpu" {
        Device::Cpu
    } else if lower.starts_with("cuda") {
        let idx = lower
            .split(':')
            .nth(1)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        if tch::Cuda::is_available() {
            Device::Cuda(idx)
        } else {
            Device::Cpu
        }
    } else {
        Device::Cpu
    }
}
