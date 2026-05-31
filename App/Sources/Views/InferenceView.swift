import CwlLlmSwiftLib
import SwiftUI

enum InferenceComparisonStatus: String, Sendable {
    case pending
    case running
    case completed
    case failed
    case cancelled

    var description: String {
        switch self {
        case .pending: "Pending"
        case .running: "Running..."
        case .completed: "Completed"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
    }
}

struct InferenceComparisonResult: Identifiable, Sendable {
    let id: String
    let engineName: String
    var status: InferenceComparisonStatus
    var timeToFirstOutputMilliseconds: Double?
    var generatedTokenCount: Int
    var totalElapsedSeconds: Double?
    var tokensPerSecond: Double?
    var errorMessage: String?
}

struct InferenceView: View {
    @State private var prompt = ""
    @State private var maximumTokenCount = 64
    @State private var temperature = 1.0

    let output: String
    let isRunningInference: Bool
    let statusMessage: String?
    let errorMessage: String?
    let selectedSourceName: String
    let canGenerate: Bool
    let selectedCompiledModelURL: URL?
    let inferenceComparisonResults: [InferenceComparisonResult]
    let isRunningInferenceComparison: Bool
    let canRunInferenceComparison: Bool
    let inferenceComparisonStatusMessage: String?
    let inferenceComparisonErrorMessage: String?
    let availableComparisonEngines: [LLMEngineDescriptor]
    @Binding var selectedInferenceComparisonEngineIDs: Set<LLMEngineIdentifier>
    @Binding var includeCompiledModelInInferenceComparison: Bool
    let generate: (String, Int, Double) -> Void
    let stop: () -> Void
    let runInferenceComparison: (String, Int, Double) -> Void
    let stopInferenceComparison: () -> Void

    var body: some View {
        TabLayout(title: "Inference", subtitle: "Generate text from the currently loaded checkpoint.") {
            VStack(alignment: .leading, spacing: 20) {
                GroupBox("Text Generation") {
                    if canGenerate {
                        VStack(alignment: .leading, spacing: 12) {
                            TextEditor(text: $prompt)
                                .frame(minHeight: 100)
                                .padding(.vertical, 8)
                                .padding(.horizontal, 4)
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary, lineWidth: 1))

                            HStack {
                                Stepper("Max tokens: \(maximumTokenCount)", value: $maximumTokenCount, in: 1...1_024)
                                Text("Temperature")
                                Slider(value: $temperature, in: 0...2)
                                Text(temperature.formatted(.number.precision(.fractionLength(2))))
                                    .monospacedDigit()
                            }

                            HStack {
                                Button("Generate") {
                                    generate(prompt, maximumTokenCount, temperature)
                                }
                                .disabled(isRunningInference || isRunningInferenceComparison)

                                Button("Stop", action: stop)
                                    .disabled(!isRunningInference)
                            }

                            Text(output.isEmpty ? "Generated output will appear here." : output)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(12)
                                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
                        }
                    } else {
                        Text("\(selectedSourceName) does not support inference yet.")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                GroupBox("Run Inference Comparison") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Menu {
                                ForEach(availableComparisonEngines) { engine in
                                    Toggle(engine.displayName, isOn: Binding(
                                        get: { selectedInferenceComparisonEngineIDs.contains(engine.id) },
                                        set: { isOn in
                                            if isOn {
                                                selectedInferenceComparisonEngineIDs.insert(engine.id)
                                            } else {
                                                selectedInferenceComparisonEngineIDs.remove(engine.id)
                                            }
                                        }
                                    ))
                                }
                                if selectedCompiledModelURL != nil {
                                    Divider()
                                    Toggle(selectedCompiledModelURL?.lastPathComponent ?? "Selected mlpackage", isOn: $includeCompiledModelInInferenceComparison)
                                }
                            } label: {
                                Label(
                                    "Sources (\(selectedSourceCount)/\(availableSourceCount))",
                                    systemImage: "shippingbox"
                                )
                            }
                            .menuStyle(.borderlessButton)
                            .fixedSize()

                            Button("Toggle All", action: toggleAllSources)
                                .disabled(isRunningInferenceComparison || availableSourceCount == 0)

                            Spacer()

                            if let selectedCompiledModelURL {
                                Text("mlpackage: \(selectedCompiledModelURL.lastPathComponent)")
                                    .foregroundStyle(.secondary)
                            }
                        }

                        HStack {
                            Button("Run comparison") {
                                runInferenceComparison(prompt, maximumTokenCount, temperature)
                            }
                            .disabled(isRunningInference || isRunningInferenceComparison || !canRunInferenceComparison)

                            Button("Cancel", action: stopInferenceComparison)
                                .disabled(!isRunningInferenceComparison)

                            if let inferenceComparisonErrorMessage, !inferenceComparisonErrorMessage.isEmpty {
                                Text(inferenceComparisonErrorMessage)
                                    .foregroundStyle(.red)
                            } else if let inferenceComparisonStatusMessage, !inferenceComparisonStatusMessage.isEmpty {
                                Text(inferenceComparisonStatusMessage)
                                    .foregroundStyle(.secondary)
                            }
                        }

                        if inferenceComparisonResults.isEmpty {
                            Text("Run a comparison to measure time-to-first-token, total generation time, and tokens per second across the selected inference sources.")
                                .foregroundStyle(.secondary)
                        } else {
                            Table(inferenceComparisonResults) {
                                TableColumn("Source") { result in
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(result.engineName)
                                        Text(result.errorMessage ?? result.status.description)
                                            .font(.caption)
                                            .foregroundStyle(color(for: result.status))
                                    }
                                }
                                TableColumn("First (ms)") { Text(formatted(result: $0.timeToFirstOutputMilliseconds)) }
                                TableColumn("Tokens") { Text("\($0.generatedTokenCount)") }
                                TableColumn("Tok / s") { Text(formatted(result: $0.tokensPerSecond)) }
                                TableColumn("Total (s)") { Text(formatted(result: $0.totalElapsedSeconds)) }
                            }
                            .frame(minHeight: 180)
                        }
                    }
                }

                if let errorMessage {
                    Text(errorMessage)
                        .foregroundStyle(.red)
                } else if let statusMessage {
                    Text(statusMessage)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var availableSourceCount: Int {
        availableComparisonEngines.count + (selectedCompiledModelURL == nil ? 0 : 1)
    }

    private var selectedSourceCount: Int {
        selectedInferenceComparisonEngineIDs.intersection(Set(availableComparisonEngines.map(\.id))).count
            + (selectedCompiledModelURL != nil && includeCompiledModelInInferenceComparison ? 1 : 0)
    }

    private func toggleAllSources() {
        let availableEngineIDs = Set(availableComparisonEngines.map(\.id))
        let allEnginesSelected = !availableEngineIDs.isEmpty && availableEngineIDs.isSubset(of: selectedInferenceComparisonEngineIDs)
        let compiledSelected = selectedCompiledModelURL == nil || includeCompiledModelInInferenceComparison
        if allEnginesSelected && compiledSelected {
            selectedInferenceComparisonEngineIDs.subtract(availableEngineIDs)
            includeCompiledModelInInferenceComparison = false
        } else {
            selectedInferenceComparisonEngineIDs.formUnion(availableEngineIDs)
            if selectedCompiledModelURL != nil {
                includeCompiledModelInInferenceComparison = true
            }
        }
    }

    private func color(for status: InferenceComparisonStatus) -> Color {
        switch status {
        case .pending: .secondary
        case .running: .accentColor
        case .completed: .green
        case .failed: .red
        case .cancelled: .orange
        }
    }

    private func formatted(result value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(3))) } ?? "—"
    }
}
