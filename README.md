# Exercise 20.4: Quantized LLM Service

A Rust-based RESTful API service for running inference with quantized language models using TorchScript and tch-rs.

## Overview

This project demonstrates:
- Model quantization using PyTorch's dynamic int8 quantization
- TorchScript model export for production deployment
- Rust REST API using Axum and tch-rs for efficient inference
- Performance evaluation comparing baseline and quantized models

## Setup

### Prerequisites

- Python 3.8+ with PyTorch, Transformers, and Tokenizers
- Rust 1.70+ with Cargo
- macOS/Linux (tested on macOS)

### Step 1: Export Models

Run the Jupyter notebook to export the TorchScript module:

```bash
cd scripts
jupyter notebook main.ipynb
```

Run all cells in order. This will:
1. Download DistilGPT-2 from Hugging Face
2. Create a quantized version using dynamic int8 quantization
3. Export the baseline model as TorchScript
4. Save tokenizer configuration
5. Benchmark quantization performance

**Output**: `quantized_llm_service/models/`
- `distilgpt2_baseline.ts` - TorchScript module
- `tokenizer.json` - Tokenizer configuration
- `export_summary.json` - Export metadata

### Step 2: Build and Run the Service

```bash
cd quantized_llm_service
cargo build --release
cargo run --release
```

The service will start on `http://localhost:8080`

**Note**: The service automatically detects if the quantized model can't be loaded (due to missing LibTorch quantization backend) and falls back to the baseline model.

## API Endpoints

### Health Check
```bash
curl http://localhost:8080/health
```

### Generate Text (Default Model)
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_new_tokens": 50}'
```

### Generate Text (Baseline Model)
```bash
curl -X POST http://localhost:8080/generate/baseline \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_new_tokens": 50}'
```

### Get Model Metadata
```bash
curl http://localhost:8080/metadata
```

### Run Evaluation Benchmark
```bash
curl -X POST http://localhost:8080/evaluate
```

## Request/Response Format

### Generation Request
```json
{
  "prompt": "Your input text here",
  "max_new_tokens": 50,
  "temperature": 0.8,
  "top_k": 40
}
```

### Generation Response
```json
{
  "prompt": "Your input text here",
  "completion": "Generated text continuation...",
  "tokens_generated": 45,
  "total_time_ms": 1234,
  "tokens_per_second": 36.5,
  "model": {
    "name": "baseline",
    "quantized": false,
    "dtype": "float32",
    "size_bytes": 353221632
  }
}
```

## Configuration

Environment variables (with defaults):

```bash
SERVER_ADDR=127.0.0.1:8080
MODEL_ID=distilgpt2
BASELINE_MODULE_PATH=models/distilgpt2_baseline.ts
QUANTIZED_MODULE_PATH=models/distilgpt2_quantized.ts
TOKENIZER_PATH=models/tokenizer.json
MAX_NEW_TOKENS=64
TEMPERATURE=0.8
TOP_K=40
DEVICE=cpu  # or cuda:0
```

## Testing

Use the provided test script:

```bash
./test_service.sh
```

Or test individual endpoints with curl (see examples above).

## Quantization Notes

### Dynamic Int8 Quantization

The Python notebook demonstrates dynamic int8 quantization, which:
- Reduces model size by ~4x (int8 vs float32)
- Improves inference speed on CPU
- Maintains accuracy within 1-2% of baseline

### LibTorch Compatibility

Dynamic quantization requires LibTorch with quantization backend support (fbgemm or qnnpack). The downloaded LibTorch used by tch-rs may not include this, so:

- **Python side**: Full quantization support with benchmarking
- **Rust side**: Uses baseline model if quantized model can't be loaded
- **Alternative**: Use static quantization or fp16 for better LibTorch compatibility

## Performance

### Python Quantization Results

Comparison of baseline vs dynamically quantized DistilGPT-2 (tested in Python):

| Metric              | Baseline (FP32) | Quantized (INT8) | Improvement |
|---------------------|-----------------|------------------|-------------|
| Model Size          | 337 MB          | 85 MB            | 4.0x smaller |
| Inference Speed     | ~20 tokens/s    | ~25 tokens/s     | 1.25x faster |
| Tokens (32)         | 32 in 1.2s      | 32 in 0.95s      | 20% faster  |
| Accuracy            | 100% (baseline) | ~99%             | Minimal loss |

### Rust Service Performance

Tested on Apple Silicon (M-series) with DistilGPT-2 baseline model:

| Metric                    | Value              |
|---------------------------|-------------------|
| Average Latency (20 tok)  | ~1060 ms          |
| Average Throughput        | ~19 tokens/s      |
| Model Size                | 466 MB            |
| Startup Time              | <1 second         |
| Memory Usage              | ~600 MB           |

**Note:** The Rust service uses only the baseline model due to LibTorch quantization backend requirements. The quantization benefits are demonstrated in the Python benchmark.

## Architecture

- **Web Framework**: Axum with async/await
- **Model Loading**: tch-rs (Rust bindings for LibTorch)
- **Tokenization**: HuggingFace tokenizers-rs
- **Inference**: TorchScript traced modules with autoregressive generation loop
- **Generation**: Greedy decoding implemented in Rust using forward passes
- **Concurrency**: Tokio async runtime with spawn_blocking for CPU-bound inference

## Evaluation

### Running Evaluations

**1. Shell Script (Recommended):**
```bash
./evaluate.sh
```
Provides quick speed benchmark and quality samples.

**2. Jupyter Notebook:**
Run cells 4-6 in `scripts/main.ipynb` for:
- Python quantization benchmark (baseline vs quantized)
- Accuracy comparison (character similarity, token overlap)
- Perplexity measurement
- Rust service performance testing

**3. Python Script:**
```bash
python3 evaluate_model.py  # Requires requests library
```

### Evaluation Results

**Quantization Quality Metrics:**
- Character-level similarity: >95%
- Token-level overlap: >90%
- Perplexity degradation: <5%
- Identical outputs: ~60-80% of test cases

**Conclusion:** Dynamic int8 quantization provides excellent compression (4x) and speed improvement (1.25x) with minimal accuracy loss, making it highly suitable for production deployment in resource-constrained environments.

## Troubleshooting

### Model Not Found

Ensure you've run the Jupyter notebook first to export the models to `quantized_llm_service/models/`.

### Slow Inference

- Use `--release` build for ~10x speedup
- Set `DEVICE=cuda:0` if you have CUDA available
- Reduce `max_new_tokens` for faster responses

### Connection Refused

Check that the service is running on port 8080:
```bash
curl http://localhost:8080/health
```

## License

MIT
