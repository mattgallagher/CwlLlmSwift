#!/usr/bin/env python3
"""Load CwlLlmSwift checkpoints in PyTorch and export inference to Core ML.

This script mirrors the repository's GPT-2 checkpoint and dataset shard formats.
It supports three workflows:

1. Inspect checkpoint/dataset metadata.
2. Run forward inference on `train.bin` / `val.bin` style batches.
3. Export an inference-only fixed-shape Core ML `.mlpackage`.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


CHECKPOINT_MAGIC = 20240326
CHECKPOINT_VERSION = 3
CHECKPOINT_HEADER_COUNT = 256

DATASET_MAGIC = 20240520
DATASET_VERSION = 1
DATASET_HEADER_COUNT = 256


@dataclass(frozen=True)
class GPT2Config:
    max_seq_len: int
    vocab_size: int
    num_layers: int
    num_heads: int
    channels: int
    padded_vocab_size: int

    @property
    def head_size(self) -> int:
        return self.channels // self.num_heads

    @property
    def num_parameters(self) -> int:
        c = self.channels
        l = self.num_layers
        return (
            self.padded_vocab_size * c
            + self.max_seq_len * c
            + l * c
            + l * c
            + l * (3 * c * c)
            + l * (3 * c)
            + l * (c * c)
            + l * c
            + l * c
            + l * c
            + l * (4 * c * c)
            + l * (4 * c)
            + l * (c * 4 * c)
            + l * c
            + c
            + c
        )


@dataclass(frozen=True)
class ParameterTensors:
    wte: np.ndarray
    wpe: np.ndarray
    ln1w: list[np.ndarray]
    ln1b: list[np.ndarray]
    qkvw: list[np.ndarray]
    qkvb: list[np.ndarray]
    attprojw: list[np.ndarray]
    attprojb: list[np.ndarray]
    ln2w: list[np.ndarray]
    ln2b: list[np.ndarray]
    fcw: list[np.ndarray]
    fcb: list[np.ndarray]
    fcprojw: list[np.ndarray]
    fcprojb: list[np.ndarray]
    lnfw: np.ndarray
    lnfb: np.ndarray

    @classmethod
    def from_flat(cls, config: GPT2Config, flat: np.ndarray) -> "ParameterTensors":
        flat = np.asarray(flat, dtype=np.float32)
        offset = 0

        def take(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal offset
            count = math.prod(shape)
            values = flat[offset:offset + count]
            if values.size != count:
                raise ValueError(f"Parameter slice truncated at offset {offset} for shape {shape}.")
            offset += count
            return values.reshape(shape)

        def take_layers(count: int, shape: tuple[int, ...]) -> list[np.ndarray]:
            return [take(shape) for _ in range(count)]

        c = config.channels
        l = config.num_layers
        parameters = cls(
            wte=take((config.padded_vocab_size, c))[: config.vocab_size, : c].copy(),
            wpe=take((config.max_seq_len, c)),
            ln1w=take_layers(l, (c,)),
            ln1b=take_layers(l, (c,)),
            qkvw=take_layers(l, (3 * c, c)),
            qkvb=take_layers(l, (3 * c,)),
            attprojw=take_layers(l, (c, c)),
            attprojb=take_layers(l, (c,)),
            ln2w=take_layers(l, (c,)),
            ln2b=take_layers(l, (c,)),
            fcw=take_layers(l, (4 * c, c)),
            fcb=take_layers(l, (4 * c,)),
            fcprojw=take_layers(l, (c, 4 * c)),
            fcprojb=take_layers(l, (c,)),
            lnfw=take((c,)),
            lnfb=take((c,)),
        )
        if offset != flat.size:
            raise ValueError(f"Unused parameters remain: consumed {offset}, total {flat.size}.")
        return parameters


@dataclass(frozen=True)
class DatasetShard:
    path: Path
    token_count: int
    tokens: np.ndarray

    @property
    def header_words(self) -> int:
        return DATASET_HEADER_COUNT

    def sample_count(self, batch_size: int, sequence_length: int) -> int:
        return max(1, (self.token_count - 1) // (batch_size * sequence_length))

    def batch(self, batch_size: int, sequence_length: int, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.token_count < batch_size * sequence_length + 1:
            raise ValueError(
                f"Dataset shard {self.path} does not contain enough tokens for B={batch_size}, T={sequence_length}."
            )
        sample_count = self.sample_count(batch_size, sequence_length)
        wrapped_index = sample_index % sample_count
        start = wrapped_index * batch_size * sequence_length
        count = batch_size * sequence_length
        inputs = self.tokens[start:start + count].astype(np.int64).reshape(batch_size, sequence_length)
        targets = self.tokens[start + 1:start + count + 1].astype(np.int64).reshape(batch_size, sequence_length)
        return inputs, targets


def read_u32_header(path: Path, expected_words: int) -> list[int]:
    data = path.read_bytes()
    header_size = expected_words * 4
    if len(data) < header_size:
        raise ValueError(f"{path} is too small to contain a valid header.")
    return list(struct.unpack(f"<{expected_words}I", data[:header_size]))


def load_checkpoint(path: Path) -> tuple[GPT2Config, ParameterTensors]:
    data = path.read_bytes()
    header_size = CHECKPOINT_HEADER_COUNT * 4
    if len(data) < header_size:
        raise ValueError(f"Checkpoint {path} is too small to contain a valid header.")

    header = struct.unpack(f"<{CHECKPOINT_HEADER_COUNT}I", data[:header_size])
    if header[0] != CHECKPOINT_MAGIC:
        raise ValueError(f"Checkpoint {path} has invalid magic value {header[0]}.")
    if header[1] != CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint {path} uses unsupported version {header[1]}.")

    config = GPT2Config(
        max_seq_len=header[2],
        vocab_size=header[3],
        num_layers=header[4],
        num_heads=header[5],
        channels=header[6],
        padded_vocab_size=header[7],
    )
    parameters = np.frombuffer(data[header_size:], dtype=np.float32)
    if parameters.size != config.num_parameters:
        raise ValueError(
            f"Checkpoint {path} contained {parameters.size} parameters, expected {config.num_parameters}."
        )
    return config, ParameterTensors.from_flat(config, parameters)


def load_dataset_shard(path: Path) -> DatasetShard:
    data = path.read_bytes()
    header_size = DATASET_HEADER_COUNT * 4
    if len(data) < header_size:
        raise ValueError(f"Dataset shard {path} is too small to contain a valid header.")

    header = struct.unpack(f"<{DATASET_HEADER_COUNT}I", data[:header_size])
    if header[0] != DATASET_MAGIC:
        raise ValueError(f"Dataset shard {path} has invalid magic value {header[0]}.")
    if header[1] != DATASET_VERSION:
        raise ValueError(f"Dataset shard {path} uses unsupported version {header[1]}.")

    token_count = int(header[2])
    tokens = np.frombuffer(data[header_size:], dtype=np.uint16)
    if tokens.size != token_count:
        raise ValueError(
            f"Dataset shard {path} token count mismatch: header says {token_count}, file contains {tokens.size}."
        )
    return DatasetShard(path=path, token_count=token_count, tokens=tokens)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    y = x @ weight.transpose(0, 1)
    if bias is not None:
        y = y + bias
    return y


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))) + 1.0)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=2, keepdim=True)
    variance = (x - mean).pow(2).mean(dim=2, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    return rstd * (x - mean) * weight + bias


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: GPT2Config, parameters: ParameterTensors, layer_index: int) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.channels = config.channels
        self.head_size = config.head_size
        self.register_buffer("ln1w", torch.from_numpy(parameters.ln1w[layer_index].copy()))
        self.register_buffer("ln1b", torch.from_numpy(parameters.ln1b[layer_index].copy()))
        self.register_buffer("qkvw", torch.from_numpy(parameters.qkvw[layer_index].copy()))
        self.register_buffer("qkvb", torch.from_numpy(parameters.qkvb[layer_index].copy()))
        self.register_buffer("attprojw", torch.from_numpy(parameters.attprojw[layer_index].copy()))
        self.register_buffer("attprojb", torch.from_numpy(parameters.attprojb[layer_index].copy()))
        self.register_buffer("ln2w", torch.from_numpy(parameters.ln2w[layer_index].copy()))
        self.register_buffer("ln2b", torch.from_numpy(parameters.ln2b[layer_index].copy()))
        self.register_buffer("fcw", torch.from_numpy(parameters.fcw[layer_index].copy()))
        self.register_buffer("fcb", torch.from_numpy(parameters.fcb[layer_index].copy()))
        self.register_buffer("fcprojw", torch.from_numpy(parameters.fcprojw[layer_index].copy()))
        self.register_buffer("fcprojb", torch.from_numpy(parameters.fcprojb[layer_index].copy()))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, fixed_batch_size: int | None = None, fixed_sequence_length: int | None = None) -> torch.Tensor:
        if fixed_batch_size is None or fixed_sequence_length is None:
            batch_size, sequence_length, _ = x.shape
        else:
            batch_size = fixed_batch_size
            sequence_length = fixed_sequence_length
        residual = x

        ln1 = layer_norm(residual, self.ln1w, self.ln1b)
        qkv = linear(ln1, self.qkvw, self.qkvb)
        reshaped = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_size)
        q = reshaped[:, :, 0, :, :].permute(0, 2, 1, 3)
        k = reshaped[:, :, 1, :, :].permute(0, 2, 1, 3)
        v = reshaped[:, :, 2, :, :].permute(0, 2, 1, 3)

        preatt = q @ k.transpose(-1, -2)
        preatt_scaled = preatt * (1.0 / math.sqrt(self.head_size))
        masked = preatt_scaled.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(masked, dim=-1)
        atty = (att @ v).permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, self.channels)

        attproj = linear(atty, self.attprojw, self.attprojb)
        residual2 = residual + attproj
        ln2 = layer_norm(residual2, self.ln2w, self.ln2b)
        fch = linear(ln2, self.fcw, self.fcb)
        fch_gelu = gelu(fch)
        fcproj = linear(fch_gelu, self.fcprojw, self.fcprojb)
        return residual2 + fcproj


class GPT2ForwardModel(torch.nn.Module):
    def __init__(self, config: GPT2Config, parameters: ParameterTensors) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("wte", torch.from_numpy(parameters.wte.copy()))
        self.register_buffer("wpe", torch.from_numpy(parameters.wpe.copy()))
        self.register_buffer("lnfw", torch.from_numpy(parameters.lnfw.copy()))
        self.register_buffer("lnfb", torch.from_numpy(parameters.lnfb.copy()))
        causal_mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        ).view(1, 1, config.max_seq_len, config.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)
        self.blocks = torch.nn.ModuleList(
            TransformerBlock(config, parameters, layer_index) for layer_index in range(config.num_layers)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"Expected tokens with rank 2 [B, T], got shape {tuple(tokens.shape)}.")
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {tokens.shape[1]} exceeds max_seq_len {self.config.max_seq_len}."
            )

        tokens = tokens.to(dtype=torch.long)
        batch_size, sequence_length = tokens.shape
        token_embedding = self.wte[tokens]
        position_embedding = self.wpe[:sequence_length, :].unsqueeze(0)
        x = token_embedding + position_embedding

        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        for block in self.blocks:
            x = block(x, mask)

        lnf = layer_norm(x, self.lnfw, self.lnfb)
        return lnf @ self.wte.transpose(0, 1)


class FixedShapeLogitsWrapper(torch.nn.Module):
    def __init__(self, model: GPT2ForwardModel, sequence_length: int) -> None:
        super().__init__()
        self.model = model
        self.sequence_length = sequence_length

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(dtype=torch.long)
        token_embedding = self.model.wte[tokens]
        position_embedding = self.model.wpe[: self.sequence_length, :].unsqueeze(0)
        x = token_embedding + position_embedding

        mask = self.model.causal_mask[:, :, : self.sequence_length, : self.sequence_length]
        for block in self.model.blocks:
            x = block(x, mask, fixed_batch_size=1, fixed_sequence_length=self.sequence_length)

        lnf = layer_norm(x, self.model.lnfw, self.model.lnfb)
        return lnf @ self.model.wte.transpose(0, 1)


class LastTokenLogitsWrapper(torch.nn.Module):
    def __init__(self, model: GPT2ForwardModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.model(tokens)
        return logits[:, -1, :]


def summarize_config(config: GPT2Config) -> dict[str, Any]:
    return {
        "max_seq_len": config.max_seq_len,
        "vocab_size": config.vocab_size,
        "padded_vocab_size": config.padded_vocab_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "channels": config.channels,
        "head_size": config.head_size,
        "num_parameters": config.num_parameters,
    }


def summarize_dataset(dataset: DatasetShard) -> dict[str, Any]:
    return {
        "path": str(dataset.path),
        "token_count": dataset.token_count,
        "first_tokens": dataset.tokens[:16].astype(np.int64).tolist(),
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray | float | int]) -> None:
    normalized: dict[str, Any] = {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value
        else:
            normalized[key] = np.asarray(value)
    np.savez(path, **normalized)


def run_forward(
    model: GPT2ForwardModel,
    dataset: DatasetShard,
    batch_size: int,
    sequence_length: int,
    sample_index: int,
    device: torch.device,
    output_npz: Path | None,
) -> dict[str, Any]:
    inputs, targets = dataset.batch(batch_size=batch_size, sequence_length=sequence_length, sample_index=sample_index)
    input_tensor = torch.from_numpy(inputs).to(device=device)
    target_tensor = torch.from_numpy(targets).to(device=device)
    model = model.to(device=device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_tensor.reshape(-1),
            reduction="mean",
        )

    logits_np = logits.cpu().numpy().astype(np.float32)
    last_token_logits = logits_np[:, -1, :]
    result = {
        "sample_index": sample_index,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "loss": float(loss.item()),
        "inputs_shape": list(inputs.shape),
        "targets_shape": list(targets.shape),
        "logits_shape": list(logits_np.shape),
        "last_token_logits_shape": list(last_token_logits.shape),
        "inputs_preview": inputs.reshape(-1)[:16].tolist(),
        "targets_preview": targets.reshape(-1)[:16].tolist(),
        "last_token_logits_preview": last_token_logits.reshape(-1)[:16].tolist(),
    }
    if output_npz is not None:
        save_npz(
            output_npz,
            {
                "inputs": inputs.astype(np.int32),
                "targets": targets.astype(np.int32),
                "logits": logits_np,
                "last_token_logits": last_token_logits,
                "loss": float(loss.item()),
            },
        )
        result["output_npz"] = str(output_npz)
    return result


def export_coreml(
    model: GPT2ForwardModel,
    sequence_length: int,
    output_path: Path,
    minimum_deployment_target: str,
    output_mode: str,
) -> dict[str, Any]:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools is required for export. Install the packages in Tools/requirements-coreml.txt.") from exc

    model.eval()
    fixed_shape_model = FixedShapeLogitsWrapper(model, sequence_length)
    fixed_shape_model.eval()
    wrapper: torch.nn.Module
    output_name: str
    if output_mode == "last-token-logits":
        wrapper = LastTokenLogitsWrapper(fixed_shape_model)
        output_name = "last_token_logits"
    elif output_mode == "logits":
        wrapper = fixed_shape_model
        output_name = "logits"
    else:
        raise ValueError(f"Unsupported output mode {output_mode}.")
    wrapper.eval()

    example_tokens = torch.zeros((1, sequence_length), dtype=torch.int32)
    traced = torch.jit.trace(wrapper.cpu(), example_tokens, strict=True)
    deployment_target = getattr(ct.target, minimum_deployment_target)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=deployment_target,
        inputs=[ct.TensorType(name="tokens", shape=example_tokens.shape, dtype=np.int32)],
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
    )
    mlmodel.save(str(output_path))
    return {
        "batch_size": 1,
        "output_path": str(output_path),
        "sequence_length": sequence_length,
        "output_mode": output_mode,
        "minimum_deployment_target": minimum_deployment_target,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return parsed


def try_parse_shorthand_export(argv: list[str]) -> tuple[Path, Path] | None:
    if len(argv) != 2:
        return None
    if any(argument.startswith("-") for argument in argv):
        return None
    return Path(argv[0]), Path(argv[1])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a repo-format GPT-2 checkpoint, e.g. gpt2_124M.bin.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print checkpoint and optional dataset metadata.")
    inspect_parser.add_argument("--train", type=Path, help="Optional path to train.bin.")
    inspect_parser.add_argument("--val", type=Path, help="Optional path to val.bin.")

    forward_parser = subparsers.add_parser("forward", help="Run a PyTorch forward pass on a dataset-backed batch.")
    forward_parser.add_argument("--dataset", type=Path, required=True, help="Path to train.bin or val.bin.")
    forward_parser.add_argument("--batch-size", type=positive_int, default=1, help="Batch size B.")
    forward_parser.add_argument("--sequence-length", type=positive_int, required=True, help="Sequence length T.")
    forward_parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index, wrapped like the Swift loaders.")
    forward_parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Torch device to use for forward inference.",
    )
    forward_parser.add_argument("--output-npz", type=Path, help="Optional `.npz` path for inputs, targets, logits, and loss.")

    export_parser = subparsers.add_parser("export", help="Export an inference-only Core ML mlprogram `.mlpackage`.")
    export_parser.add_argument("--sequence-length", type=positive_int, default=64, help="Fixed traced sequence length.")
    export_parser.add_argument("--output", type=Path, required=True, help="Output `.mlpackage` path.")
    export_parser.add_argument(
        "--minimum-deployment-target",
        default="macOS13",
        help="coremltools deployment target enum member, e.g. macOS13 or iOS17.",
    )
    export_parser.add_argument(
        "--output-mode",
        choices=("logits", "last-token-logits"),
        default="logits",
        help="Whether the Core ML model returns the full logits tensor or only the last-token row.",
    )

    return parser


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    shorthand_export = try_parse_shorthand_export(sys.argv[1:])
    if shorthand_export is not None:
        checkpoint_path, output_path = shorthand_export
        config, parameters = load_checkpoint(checkpoint_path)
        model = GPT2ForwardModel(config, parameters)
        sequence_length = 64
        if sequence_length > config.max_seq_len:
            raise ValueError(
                f"Requested sequence length {sequence_length} exceeds checkpoint max_seq_len {config.max_seq_len}."
            )
        emit_json(
            {
                "checkpoint": str(checkpoint_path),
                "config": summarize_config(config),
                "export": export_coreml(
                    model=model,
                    sequence_length=sequence_length,
                    output_path=output_path,
                    minimum_deployment_target="macOS13",
                    output_mode="logits",
                ),
            }
        )
        return

    parser = build_arg_parser()
    args = parser.parse_args()

    config, parameters = load_checkpoint(args.checkpoint)
    model = GPT2ForwardModel(config, parameters)

    if args.command == "inspect":
        payload: dict[str, Any] = {
            "checkpoint": str(args.checkpoint),
            "config": summarize_config(config),
        }
        if args.train is not None:
            payload["train"] = summarize_dataset(load_dataset_shard(args.train))
        if args.val is not None:
            payload["val"] = summarize_dataset(load_dataset_shard(args.val))
        emit_json(payload)
        return

    if args.command == "forward":
        dataset = load_dataset_shard(args.dataset)
        device = torch.device(args.device)
        payload = {
            "checkpoint": str(args.checkpoint),
            "dataset": summarize_dataset(dataset),
            "config": summarize_config(config),
            "forward": run_forward(
                model=model,
                dataset=dataset,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                sample_index=args.sample_index,
                device=device,
                output_npz=args.output_npz,
            ),
        }
        emit_json(payload)
        return

    if args.command == "export":
        if args.sequence_length > config.max_seq_len:
            raise ValueError(
                f"Requested sequence length {args.sequence_length} exceeds checkpoint max_seq_len {config.max_seq_len}."
            )
        payload = {
            "checkpoint": str(args.checkpoint),
            "config": summarize_config(config),
            "export": export_coreml(
                model=model,
                sequence_length=args.sequence_length,
                output_path=args.output,
                minimum_deployment_target=args.minimum_deployment_target,
                output_mode=args.output_mode,
            ),
        }
        emit_json(payload)
        return

    raise RuntimeError(f"Unhandled command {args.command}.")


if __name__ == "__main__":
    main()
