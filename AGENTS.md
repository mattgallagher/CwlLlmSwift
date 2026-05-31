# AGENTS.md

## Project Goal

This repository will contain a series of implementations of Andrej Karpathy's `llm.c` in Swift, plus a shared SwiftUI test harness for exercising them.

The engine targets are:

1. Basic Swift
2. Fast Swift
3. Multithreaded Swift
4. Accelerate BLAS — `cblas_sgemm`-accelerated matmul
5. Accelerate BNNS — compiled BNNS graph forward pass
6. MPSGraph
7. MLTensor
8. Metal

In addition, the repository vendors Karpathy's `train_gpt2.c` as a C reference implementation for validation purposes. It is treated as a ninth reference backend rather than a primary Swift target. It was the first implementation brought into the repository and acts as the numerical baseline for all Swift backends.

The intent is to keep the different implementations comparable, educational, and close in spirit to `llm.c`, while still respecting Swift and Apple platform conventions.

## Current State

The SwiftUI document-based harness is functional end-to-end. All nine backends are fully implemented (C Reference, Raw Swift, Fast Swift, Multi-Threaded Swift, BLAS, BNNS, MPSGraph, MLTensor, Metal) with training, inference, checkpoint save/load, and cross-implementation validation tests.

Project generation uses `xcodegen`. The checked-in project definition lives in `project.yml`, and the `.xcodeproj` is generated from it rather than edited manually.

The target toolchain is Xcode 26.0 on macOS with Swift 6.0. Code is valid under Swift 6 strict concurrency checking, using explicit unsafe annotations only where they are truly needed for performance or low-level interoperability.

## Repository Structure

- `App/Sources/` — SwiftUI app entry point and views (ContentView, TrainingView, InferenceView, DatasetImportView, DocumentTab)
- `CwlLlmSwiftLib/Sources/CwlLlmSwiftLib/` — Core library: engine protocols, registry, document model, shared types, and all engine implementations
- `CwlLlmSwiftLib/Sources/CLLMCReference/` — Vendored C reference (`train_gpt2.c`) adapted for library-style use
- `CwlLlmSwiftLib/Tests/CwlLlmSwiftLibTests/` — Test suite including cross-implementation validation
- `project.yml` — XCodegen project definition

## Immediate Priority

The next priorities are:

1. Harness refinement — improved metrics presentation, error handling, capability-aware UI states, documentation
2. Harness refinement — improved metrics presentation, error handling, capability-aware UI states, documentation

It should support:

1. Selecting a training input file
2. Selecting the training engine
3. Starting and monitoring training
4. Running inference against the current model
5. Saving and reloading model state

## Harness Requirements

### 1. Dataset Selection

The app must allow the user to choose a prepared training input file from disk.

We will not commit training datasets to the repository.

The repository should instead include documentation describing how to:

1. Download supported datasets such as `tinyshakespeare` and `openwebtext`
2. Prepare/tokenize them using Karpathy's `nanoGPT` dataset scripts
3. Place or select the resulting files for use by this app

The harness should be designed so dataset handling is not tightly coupled to any one training backend.

### 2. Engine Selection

The app must provide a clear way to choose between the available training/inference engines.

Engine list:

1. C Reference (numerical baseline)
2. Raw Swift
3. Accelerate BLAS
4. Accelerate BNNS
5. MPSGraph
6. MLTensor
7. Metal

The UI and code structure allow engines to be stubbed or unavailable without blocking the harness. The `LLMEngineRegistry` pre-registers all engines with capability descriptors.

### 3. Training

The harness must be able to run training and display live statistics, including at minimum:

1. Current iteration or step
2. Training iterations per second
3. Current training loss when available
4. Current validation loss when available

Training should be structured so the UI remains responsive and receives incremental progress updates.

### 4. Inference

The harness must support text generation against the currently loaded model state.

Inference UI should allow:

1. Entering a prompt
2. Running generation
3. Viewing generated output

Inference should use the same selected engine where practical, but the architecture should allow an engine to support training, inference, or both.

### 5. Persistence

The harness must support saving the current trained model state and loading it later.

Saved state should be usable for:

1. Resuming training
2. Running inference later

When possible, keep persistence formats explicit and versionable.

### 6. Cross-Implementation Validation

The repository includes tests that validate implementations against each other.

The test suite (`LLMEngineSyntheticComparisonTests`) compares equivalent operations across implementations for:

1. Forward inference
2. Back propagation
3. Full training steps
4. Checkpoint round-trip consistency

These comparisons use bounded numerical tolerances that vary by engine (Raw Swift: 1e-5, BLAS: 5e-4, BNNS: 5e-3).

The test architecture compares:

1. Each Swift backend against the C reference implementation
## Architectural Guidance

The codebase uses a shared abstraction for model lifecycle operations, with concrete backends per framework.

Implemented shared concepts:

1. `LLMEngine` protocol for training/inference with `AsyncThrowingStream` progress
2. `LLMEngineCapabilities` for capability reporting (training, inference, checkpointing)
3. `LLMTrainingProgress` for step, timing, and loss reporting
4. `LLMInferenceRequest` / `LLMInferenceResponse` / `LLMInferenceChunk` for generation
5. `LLMCheckpointCodec` / `LLMCheckpointDescriptor` for checkpoint save/load
7. `LLMEngineRegistry` for engine discovery and selection

Each engine runtime is implemented as a Swift actor (e.g. `LLMSwiftRuntime`, `LLMCReferenceRuntime`, `LLMBlasRuntime`, `LLMBNNSRuntime`).

Prefer small, explicit protocols and types over deeply abstracted designs.

Keep the harness usable even while some engines are stubbed.

## Scope Guidance

Implementations optimize for correctness, inspectability, and comparability to `llm.c`, not for maximum performance.

The vendored C reference backend serves as the numerical baseline. The Raw Swift backend is the primary Swift baseline.

The BLAS backend demonstrates `cblas_sgemm` acceleration for matmul while keeping other operations in plain Swift. The BNNS backend demonstrates compiled BNNS graph execution for the forward pass.

New backends should aim to preserve equivalent model behavior and similar training/inference semantics wherever feasible. Each new backend should add matching cross-implementation validation tests as part of its bring-up.

## Documentation Expectations

This repository should eventually include:

1. Setup instructions for Apple platform toolchains and dependencies
2. Dataset download and preparation instructions using `nanoGPT`
3. Notes about differences between the Swift backends
4. Any backend-specific limitations or platform requirements
5. Documentation for cross-implementation validation strategy and tolerance expectations

## Editing Guidance For Future Agents

When implementing features in this repository:

1. Preserve comparability across engine implementations
2. Prefer minimal abstractions until repeated patterns are real
3. Keep UI code separate from engine internals where practical
4. Avoid coupling dataset preparation to app code; document it instead
5. Make partial implementations visible in the harness rather than hiding them behind incomplete abstractions
6. Prefer testable boundaries that support numerical comparison across implementations
7. Keep the vendored C reference as close as practical to upstream structure while adapting it for library-style use

## Build Guidance

When building the app from the command line, prefer piping `xcodebuild` through `xcsift -w` to keep Xcode output readable while preserving warnings and failures. Generally prefer `Release` configuration for the app since the code is too slow in `Debug` configuration.

Example:

`xcodebuild -scheme CwlLlmSwift -project CwlLlmSwift.xcodeproj -configuration Release build | xcsift -w`

Run the full test suite through Xcode rather than `swift test`, so package resources such as the Metal backend's compiled `default.metallib` are available at runtime.

Example:

`xcodebuild -scheme CwlLlmSwift -project CwlLlmSwift.xcodeproj -configuration Debug test | xcsift -w`

Datasets are kept small for tests so `Debug` should be acceptable but larger non-synthetic tests may require `Release` configuration.

## Non-Goals For Current Phase

Current priorities do not include:

1. Bundling datasets in the repository
2. In-app tokenization (dataset preparation remains external via `nanoGPT`)
3. Maximizing training speed before all backends are correct and comparable
4. Reproducing every feature of `llm.c` or `nanoGPT`

Cross-implementation validation is an ongoing requirement and should not be deferred behind full engine parity.
