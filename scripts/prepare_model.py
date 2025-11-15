#!/usr/bin/env python3
"""Utility script to export baseline and int8-quantized TorchScript LLM modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="distilgpt2",
        help="🤗 Hub model identifier (default: distilgpt2)",
    )
    parser.add_argument(
        "--output-dir",
        default="../quantized_llm_service/models",
        help="Directory where TorchScript modules and tokenizer assets will be stored.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Default max tokens baked into scripted generator (can be overridden at runtime).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Default top-k filtering threshold used by scripted sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Default temperature used by scripted sampling.",
    )
    return parser.parse_args()


class ScriptableGenerator(torch.nn.Module):
    """Minimal wrapper exposing a TorchScript-friendly generate method."""

    def __init__(
        self,
        model: torch.nn.Module,
        default_max_new_tokens: int,
        default_temperature: float,
        default_top_k: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.default_top_k = default_top_k

    @torch.jit.export
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")

        max_new = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        temp = temperature if temperature is not None else self.default_temperature
        topk = top_k if top_k is not None else self.default_top_k

        tokens = input_ids
        past_key_values = None
        for _ in range(max_new):
            outputs = self.model(
                input_ids=tokens[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]

            if temp != 1.0:
                logits = logits / temp

            if topk > 0:
                topk = min(topk, logits.size(-1))
                values, _ = torch.topk(logits, topk)
                kth = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            past_key_values = outputs.past_key_values
        return tokens


def script_and_save(model: torch.nn.Module, args: argparse.Namespace, suffix: str, output_dir: Path) -> Path:
    generator = ScriptableGenerator(
        model,
        default_max_new_tokens=args.max_new_tokens,
        default_temperature=args.temperature,
        default_top_k=args.top_k,
    )
    scripted = torch.jit.script(generator)
    target_path = output_dir / f"{args.model_id}_{suffix}.ts"
    scripted.save(str(target_path))
    return target_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(output_dir)

    print(f"Loading baseline model {args.model_id}...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(args.model_id)
    model_fp32.eval()

    print("Scripting baseline module...")
    baseline_path = script_and_save(model_fp32, args, "baseline", output_dir)

    print("Applying dynamic int8 quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_model.eval()

    print("Scripting quantized module...")
    quantized_path = script_and_save(quantized_model, args, "quantized", output_dir)

    summary = {
        "baseline_module": str(baseline_path),
        "quantized_module": str(quantized_path),
        "tokenizer_dir": str(output_dir),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))

    for name, path in summary.items():
        if "module" in name:
            size = Path(path).stat().st_size / (1024 * 1024)
            print(f"{name}: {path} ({size:.2f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
