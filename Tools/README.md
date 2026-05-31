# Core ML Export Tooling

`llm_coreml_export.py` loads this repository's GPT-2 checkpoint and dataset shard formats directly, runs the forward graph in PyTorch, and exports an inference-only Core ML `mlprogram` as a `.mlpackage`.

## Install

```bash
$(brew --prefix python@3.12)/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Tools/requirements-coreml.txt
```

## Inspect Assets

```bash
python Tools/llm_coreml_export.py \
  --checkpoint gpt2_124M.bin \
  inspect \
  --train train.bin \
  --val val.bin
```

## Run Forward On A Dataset Batch

This uses the same batch slicing as the Swift runtimes: `sample_index * (B * T)` into the token stream, with `inputs` and one-token-shifted `targets`.

```bash
python Tools/llm_coreml_export.py \
  --checkpoint gpt2_124M.bin \
  forward \
  --dataset train.bin \
  --batch-size 1 \
  --sequence-length 64 \
  --sample-index 0 \
  --output-npz /tmp/tinyshakespeare-forward.npz
```

The `.npz` file contains:

- `inputs`
- `targets`
- `logits`
- `last_token_logits`
- `loss`

## Export Core ML `.mlpackage`

The exporter is fixed-shape and inference-only. It exports `B=1` models because the app uses these packages for single-prompt generation and inference comparison.

For the common case, use the shorthand form:

```bash
python Tools/llm_coreml_export.py gpt2_124M.bin gpt2_124M_T64.mlpackage
```

That uses the default export settings: sequence length `64`, output mode `logits`, and minimum deployment target `macOS13`.

The explicit form is still available when you need to override defaults:

```bash
python Tools/llm_coreml_export.py \
  --checkpoint gpt2_124M.bin \
  export \
  --output /tmp/gpt2_124M_T64.mlpackage \
  --minimum-deployment-target macOS13
```

Use `--output-mode last-token-logits` if the Swift runtime only needs the next-token row.

## Notes

- Checkpoint parsing matches `LLMCheckpointCodec`.
- Dataset parsing matches the Swift token loaders used by the engines.
- The graph mirrors the existing forward pass: embedding, causal self-attention, MLP, final layer norm, tied output projection.
- The exporter currently traces a fixed `B=1` shape. That is the simplest starting point for integrating the compiled Core ML model into the app's inference loop.
