# AGENTS.md

## Project Goal

This branch captures phase 1 of the repository: a SwiftUI harness plus the first six GPT-2 backends.

The engines on this branch are:

1. `llm.c` C reference validation engine
2. Basic Swift
3. Fast Swift
4. Multithreaded Swift
5. Direct AMX
6. Metal

The C reference backend remains the numerical baseline for the Swift implementations.

## Current State

The SwiftUI document-based harness supports training, inference, checkpoint save/load, and cross-implementation validation for the phase-1 engines listed above.

Project generation uses `xcodegen`. The checked-in project definition lives in `project.yml`, and the `.xcodeproj` is generated from it rather than edited manually.

The target toolchain is Xcode 26.0 on macOS with Swift 6.0. Code is valid under Swift 6 strict concurrency checking, using explicit unsafe annotations only where they are truly needed for performance or low-level interoperability.

## Repository Structure

- `App/Sources/` — SwiftUI app entry point and views
- `CwlLlmSwiftLib/Sources/CwlLlmSwiftLib/` — Core library, shared types, and the phase-1 engine implementations
- `CwlLlmSwiftLib/Sources/CLLMCReference/` — Vendored C reference implementation adapted for library-style use
- `CwlLlmSwiftLib/Tests/CwlLlmSwiftLibTests/` — Test suite including cross-implementation validation
- `project.yml` — XcodeGen project definition

## Harness Requirements

The app should support:

1. Selecting a training input file
2. Selecting a training engine
3. Starting and monitoring training
4. Running inference against the current model
5. Saving and reloading model state

Dataset preparation remains external. Do not bundle datasets in the repository.

## Architectural Guidance

Implemented shared concepts include:

1. `LLMEngine` protocol for training and inference with `AsyncThrowingStream` progress
2. `LLMEngineCapabilities` for capability reporting
3. `LLMTrainingProgress` for step, timing, and loss reporting
4. `LLMInferenceRequest` / `LLMInferenceResponse` / `LLMInferenceChunk` for generation
5. `LLMCheckpointCodec` / `LLMCheckpointDescriptor` for checkpoint save/load
6. `LLMEngineRegistry` for engine discovery and selection

Each engine runtime is implemented as a Swift actor where practical. Prefer small, explicit protocols and types over deeper abstraction.

## Scope Guidance

This branch intentionally excludes later-phase work such as BLAS, BNNS, MPSGraph, Core ML, MLXSwift, and the Core ML export tooling.

Implementations optimize for correctness, inspectability, and comparability to `llm.c`, not for maximum performance.

## Editing Guidance For Future Agents

When implementing features in this repository:

1. Preserve comparability across the phase-1 engine implementations
2. Prefer minimal abstractions until repeated patterns are real
3. Keep UI code separate from engine internals where practical
4. Avoid coupling dataset preparation to app code; document it instead
5. Prefer testable boundaries that support numerical comparison across implementations
6. Keep the vendored C reference as close as practical to upstream structure while adapting it for library-style use

## Build Guidance

When building the app from the command line, prefer piping `xcodebuild` through `xcsift -w` to keep Xcode output readable while preserving warnings and failures.

Example:

`xcodebuild -scheme CwlLlmSwift -project CwlLlmSwift.xcodeproj -configuration Debug build | xcsift -w`
