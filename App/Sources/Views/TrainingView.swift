import CwlLlmSwiftLib
import SwiftUI

struct TrainingView: View {
    let availableCheckpoints: [LLMCheckpointDescriptor]
    @Binding var selectedCheckpointID: String
    @Binding var stepCount: Int
    @Binding var batchSize: Int
    @Binding var sequenceLength: Int
    @Binding var learningRate: Double
    @Binding var validationBatchCount: Int
    @Binding var manualCheckpointName: String
    let canCreateManualCheckpoint: Bool
    let createManualCheckpoint: () -> Void
    let progress: LLMTrainingProgress
    let timeToFirstOutputMilliseconds: Double?
    let isRunningTraining: Bool
    let isStoppingTraining: Bool
    let statusMessage: String?
    let errorMessage: String?
    let selectedEngine: (any LLMEngine)?
    let startTraining: () -> Void
    let stopTraining: () -> Void
    let loadCheckpoint: () -> Void

    var body: some View {
        TabLayout(title: "Training", subtitle: selectedEngine?.descriptor.summary) {
            VStack(alignment: .leading, spacing: 12) {
                if let engine = selectedEngine, engine.descriptor.capabilities.contains(.training) {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack(alignment: .top) {
                            VStack {
                                GroupBox("Training parameters") {
                                    HStack {
                                        Stepper("Steps: \(stepCount)", value: $stepCount, in: 1...10_000)
                                        Stepper("Batch: \(batchSize)", value: $batchSize, in: 1...64)
                                    }
                                    HStack {
                                        Stepper("Sequence: \(sequenceLength)", value: $sequenceLength, in: 8...1_024, step: 8)
                                        Stepper("Validation batches: \(validationBatchCount)", value: $validationBatchCount, in: 1...64)
                                    }
                                    HStack {
                                        Text("Learning rate")
                                        TextField("Learning rate", value: $learningRate, format: .number)
                                            .textFieldStyle(.roundedBorder)
                                            .frame(maxWidth: 180)
                                    }
                                    .frame(maxWidth: .infinity)
                                }
                                
                                HStack {
                                    Button("Start Training", action: startTraining)
                                        .disabled(isRunningTraining)
                                    Button(isStoppingTraining ? "Stopping..." : "Stop", action: stopTraining)
                                        .disabled(!isRunningTraining || isStoppingTraining)
                                }
                                
                                if let errorMessage, !errorMessage.isEmpty {
                                    Text(errorMessage)
                                        .foregroundStyle(.red)
                                } else {
                                    Text(statusMessage ?? "")
                                        .foregroundStyle(.secondary)
                                }
                            }

                            GroupBox("Progress") {
                                LabeledContent("Current step") { Text("\(progress.step)") }
                                LabeledContent("Time to first output (ms)") { Text(formattedMilliseconds(timeToFirstOutputMilliseconds)) }
                                LabeledContent("Iterations / second") { Text(formattedRate(progress.iterationsPerSecond)) }
                                LabeledContent("Forward pass (ms)") { Text(formattedMilliseconds(progress.forwardPassMilliseconds)) }
                                LabeledContent("Backward pass (ms)") { Text(formattedMilliseconds(progress.backwardPassMilliseconds)) }
                                LabeledContent("Training loss") { Text(formattedLoss(progress.trainingLoss)) }
                                LabeledContent("Validation loss") { Text(formattedLoss(progress.validationLoss)) }
                                    .frame(maxWidth: .infinity)
                            }
                        }
                    }
                } else {
                    Text("\(selectedEngine?.descriptor.displayName ?? "Selected engine") does not support training yet.")
                        .foregroundStyle(.secondary)
                }

                GroupBox("Checkpoints") {
                    HStack {
                        TextField("Manual checkpoint name", text: $manualCheckpointName)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 200)
                        Button("Create Manual Checkpoint", action: createManualCheckpoint)
                            .disabled(!canCreateManualCheckpoint)
                    }

                    HStack {
                        Picker("Restore checkpoint", selection: $selectedCheckpointID) {
                            Text("Select a checkpoint").tag("")
                            ForEach(availableCheckpoints) { checkpoint in
                                Text("\(checkpoint.name) [\(checkpoint.kind.rawValue)]").tag(checkpoint.id)
                            }
                        }
                        Button("Restore", action: loadCheckpoint)
                            .disabled(selectedCheckpointID.isEmpty)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
    }

    private func formattedRate(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(3))) } ?? "Pending"
    }

    private func formattedMilliseconds(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(3))) } ?? "Pending"
    }

    private func formattedLoss(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(6))) } ?? "Pending"
    }
}
