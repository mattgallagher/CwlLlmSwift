import Charts
import CwlLlmSwiftLib
import SwiftUI

enum ComparisonEngineStatus: String, Sendable {
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

struct ComparisonLossSample: Identifiable, Hashable, Sendable {
    let id = UUID()
    let engineName: String
    let time: Double
    let loss: Double
}

struct ComparisonEngineResult: Identifiable, Sendable {
    let id: LLMEngineIdentifier
    let engineName: String
    var status: ComparisonEngineStatus
    var progress: LLMTrainingProgress
    var timeToFirstOutputMilliseconds: Double?
    var totalElapsedSeconds: Double?
    var errorMessage: String?

    var averageIterationsPerSecond: Double? {
        guard progress.step > 1,
              let totalElapsedSeconds,
              let timeToFirstOutputMilliseconds else {
            return nil
        }
        let measuredSeconds = totalElapsedSeconds - timeToFirstOutputMilliseconds / 1_000
        guard measuredSeconds > 0 else {
            return nil
        }
        return Double(progress.step - 1) / measuredSeconds
    }
}

struct ComparisonView: View {
    let results: [ComparisonEngineResult]
    let lossSamples: [ComparisonLossSample]
    let isRunning: Bool
    let canRun: Bool
    @Binding var stepCount: Int
    let availableEngines: [LLMEngineDescriptor]
    @Binding var selectedEngineIDs: Set<LLMEngineIdentifier>
    let statusMessage: String?
    let errorMessage: String?
    let runComparison: () -> Void
    let cancelComparison: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Button("Run comparison", action: runComparison)
                    .disabled(isRunning || !canRun || selectedEngineIDs.isEmpty)
                Button("Cancel", action: cancelComparison)
                    .disabled(!isRunning)
                Stepper("Steps: \(stepCount)", value: $stepCount, in: 1...10_000)
                    .disabled(isRunning)
                    .fixedSize()
                Menu {
                    ForEach(availableEngines) { engine in
                        Toggle(engine.displayName, isOn: Binding(
                            get: { selectedEngineIDs.contains(engine.id) },
                            set: { isOn in
                                if isOn {
                                    selectedEngineIDs.insert(engine.id)
                                } else {
                                    selectedEngineIDs.remove(engine.id)
                                }
                            }
                        ))
                    }
                } label: {
                    Label(
                        "Engines (\(selectedEngineIDs.count)/\(availableEngines.count))",
                        systemImage: "shippingbox"
                    )
                }
                .menuStyle(.borderlessButton)
                .disabled(isRunning || availableEngines.isEmpty)
                .fixedSize()
                Button("Toggle All", action: toggleAllEngines)
                    .disabled(isRunning || availableEngines.isEmpty)

                Group {
                    if !canRun {
                        Text("Comparison requires dataset files and an initial checkpoint.")
                            .foregroundStyle(.secondary)
                    } else if let errorMessage, !errorMessage.isEmpty {
                        Text(errorMessage).foregroundStyle(.red)
                    } else if let statusMessage, !statusMessage.isEmpty {
                        Text(statusMessage).foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            GroupBox("Training loss over time") {
                if lossSamples.isEmpty {
                    Text("Run a comparison to plot per-engine training loss against wall-clock time.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, minHeight: 260)
                } else {
                    Chart(lossSamples) { sample in
                        LineMark(
                            x: .value("Elapsed time (s)", sample.time),
                            y: .value("Training loss", sample.loss)
                        )
                        .foregroundStyle(by: .value("Engine", sample.engineName))
                        .interpolationMethod(.linear)
                    }
                    .chartYScale(domain: lossYDomain)
                    .chartXAxisLabel("Elapsed time (s)")
                    .chartYAxisLabel("Training loss")
                    .chartLegend(position: .bottom, alignment: .leading)
                    .frame(minHeight: 260)
                }
            }

            GroupBox("Per-engine training statistics") {
                if results.isEmpty {
                    Text("No results yet.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Table(results) {
                        TableColumn("Engine") { result in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(result.engineName)
                                Text(result.errorMessage ?? result.status.description)
                                    .font(.caption)
                                    .foregroundStyle(color(for: result.status))
                            }
                        }
                        TableColumn("Step") { Text("\($0.progress.step)") }
                        TableColumn("First (ms)") { Text(formattedMilliseconds($0.timeToFirstOutputMilliseconds)) }
                        TableColumn("Avg iter / s") { Text(formattedRate($0.averageIterationsPerSecond)) }
                        TableColumn("Fwd (ms)") { Text(formattedMilliseconds($0.progress.forwardPassMilliseconds)) }
                        TableColumn("Bwd (ms)") { Text(formattedMilliseconds($0.progress.backwardPassMilliseconds)) }
                        TableColumn("Train loss") { Text(formattedLoss($0.progress.trainingLoss)) }
                        TableColumn("Val loss") { Text(formattedLoss($0.progress.validationLoss)) }
                        TableColumn("Total (s)") { Text(formattedSeconds($0.totalElapsedSeconds)) }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(maxHeight: .infinity)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var lossYDomain: ClosedRange<Double> {
        let losses = lossSamples.map(\.loss)
        guard let minLoss = losses.min(), let maxLoss = losses.max() else {
            return 0...1
        }
        guard minLoss < maxLoss else {
            let pad = max(abs(minLoss) * 0.05, 0.01)
            return (minLoss - pad)...(maxLoss + pad)
        }
        let pad = (maxLoss - minLoss) * 0.05
        return (minLoss - pad)...(maxLoss + pad)
    }

    private func toggleAllEngines() {
        let availableEngineIDs = Set(availableEngines.map(\.id))
        if !availableEngineIDs.isEmpty && availableEngineIDs.isSubset(of: selectedEngineIDs) {
            selectedEngineIDs.subtract(availableEngineIDs)
        } else {
            selectedEngineIDs.formUnion(availableEngineIDs)
        }
    }

    private func color(for status: ComparisonEngineStatus) -> Color {
        switch status {
        case .pending: .secondary
        case .running: .accentColor
        case .completed: .green
        case .failed: .red
        case .cancelled: .orange
        }
    }

    private func formattedRate(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(3))) } ?? "—"
    }

    private func formattedMilliseconds(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(3))) } ?? "—"
    }

    private func formattedSeconds(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(3))) } ?? "—"
    }

    private func formattedLoss(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(6))) } ?? "—"
    }
}
