import CwlLlmSwiftLib
import SwiftUI

struct ContentView: View {
    @Binding var document: LLMDatasetDocument

    @State private var selectedTab: DocumentTab = .training
    @State private var selectedEngineID: LLMEngineIdentifier = .cReference
    @State private var trainingStepCount = 10
    @State private var trainingBatchSize = 4
    @State private var trainingSequenceLength = 64
    @State private var trainingLearningRate = 1e-4
    @State private var validationBatchCount = 1
    @State private var inferenceOutput = ""
    @State private var selectedCheckpointID = ""
    @State private var manualCheckpointName = ""
    @State private var isRunningTraining = false
    @State private var isStoppingTraining = false
    @State private var isRunningInference = false
    @State private var statusMessage: String?
    @State private var errorMessage: String?
    @State private var trainingTask: Task<Void, Never>?
    @State private var inferenceTask: Task<Void, Never>?
    @State private var timeToFirstOutputMilliseconds: Double?
    @State private var comparisonResults: [ComparisonEngineResult] = []
    @State private var comparisonLossSamples: [ComparisonLossSample] = []
    @State private var isRunningComparison = false
    @State private var comparisonTask: Task<Void, Never>?
    @State private var comparisonStatusMessage: String?
    @State private var comparisonErrorMessage: String?
    @State private var comparisonStepCount = 20
    @State private var selectedComparisonEngineIDs: Set<LLMEngineIdentifier>
    @State private var inferenceComparisonResults: [InferenceComparisonResult] = []
    @State private var isRunningInferenceComparison = false
    @State private var inferenceComparisonTask: Task<Void, Never>?
    @State private var inferenceComparisonStatusMessage: String?
    @State private var inferenceComparisonErrorMessage: String?
    @State private var selectedInferenceComparisonEngineIDs: Set<LLMEngineIdentifier>

    init(document: Binding<LLMDatasetDocument>) {
        self._document = document
        self._selectedComparisonEngineIDs = State(initialValue: Set(
            document.wrappedValue.engineRegistry.descriptors
                .filter { $0.capabilities.contains(.training) && $0.isAvailable }
                .map(\.id)
        ))
        self._selectedInferenceComparisonEngineIDs = State(initialValue: Set(
            document.wrappedValue.engineRegistry.descriptors
                .filter { $0.capabilities.contains(.inference) && $0.isAvailable }
                .map(\.id)
        ))
    }

    private var engineRegistry: LLMEngineRegistry { document.engineRegistry }

    private var comparisonAvailableEngines: [LLMEngineDescriptor] {
        engineRegistry.descriptors.filter { $0.capabilities.contains(.training) && $0.isAvailable }
    }

    private var inferenceAvailableEngines: [LLMEngineDescriptor] {
        engineRegistry.descriptors.filter { $0.capabilities.contains(.inference) && $0.isAvailable }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            TabContainer {
                if document.isImportingNewDataset {
                    DatasetImportView(document: $document)
                } else {
                    Group {
                        switch selectedTab {
                        case .training:
                            TrainingView(
                                availableCheckpoints: document.availableCheckpoints,
                                selectedCheckpointID: $selectedCheckpointID,
                                stepCount: $trainingStepCount,
                                batchSize: $trainingBatchSize,
                                sequenceLength: $trainingSequenceLength,
                                learningRate: $trainingLearningRate,
                                validationBatchCount: $validationBatchCount,
                                manualCheckpointName: $manualCheckpointName,
                                canCreateManualCheckpoint: document.currentCheckpointData != nil,
                                createManualCheckpoint: createManualCheckpoint,
                                progress: document.trainingProgress,
                                timeToFirstOutputMilliseconds: timeToFirstOutputMilliseconds,
                                isRunningTraining: isRunningTraining,
                                isStoppingTraining: isStoppingTraining,
                                statusMessage: statusMessage,
                                errorMessage: errorMessage,
                                selectedEngine: selectedEngine,
                                startTraining: startTraining,
                                stopTraining: stopTraining,
                                loadCheckpoint: loadCheckpoint
                            )
                        case .inference:
                            InferenceView(
                                output: inferenceOutput,
                                isRunningInference: isRunningInference,
                                statusMessage: statusMessage,
                                errorMessage: errorMessage,
                                selectedSourceName: selectedInferenceSourceName,
                                canGenerate: selectedInferenceCanGenerate,
                                inferenceComparisonResults: inferenceComparisonResults,
                                isRunningInferenceComparison: isRunningInferenceComparison,
                                canRunInferenceComparison: canRunInferenceComparison,
                                inferenceComparisonStatusMessage: inferenceComparisonStatusMessage,
                                inferenceComparisonErrorMessage: inferenceComparisonErrorMessage,
                                availableComparisonEngines: inferenceAvailableEngines,
                                selectedInferenceComparisonEngineIDs: $selectedInferenceComparisonEngineIDs,
                                generate: runInference,
                                stop: stopInference,
                                runInferenceComparison: startInferenceComparison,
                                stopInferenceComparison: stopInferenceComparison
                            )
                        case .comparison:
                            ComparisonView(
                                results: comparisonResults,
                                lossSamples: comparisonLossSamples,
                                isRunning: isRunningComparison,
                                canRun: document.trainData != nil
                                    && document.valData != nil
                                    && document.initialCheckpointData != nil,
                                stepCount: $comparisonStepCount,
                                availableEngines: comparisonAvailableEngines,
                                selectedEngineIDs: $selectedComparisonEngineIDs,
                                statusMessage: comparisonStatusMessage,
                                errorMessage: comparisonErrorMessage,
                                runComparison: startComparison,
                                cancelComparison: stopComparison
                            )
                        }
                    }
                    .onAppear(perform: initializeCheckpointSelection)
                    .onChange(of: document.currentCheckpoint?.id) { _, _ in
                        initializeCheckpointSelection()
                    }
                }
            }
        }
        .frame(minWidth: 700, minHeight: 560)
        .toolbar {
            if !document.isImportingNewDataset {
                ToolbarItem(placement: .principal) {
                    ControlGroup {
                        ForEach(DocumentTab.allCases) { section in
                            Toggle(
                                isOn: Binding(
                                    get: { selectedTab == section },
                                    set: { isOn in
                                        if isOn {
                                            selectedTab = section
                                        }
                                    }
                                )
                            ) {
                                Label {
                                    Text(section.title)
                                } icon: {
                                    Image(systemName: section.systemImage)
                                }
                                .labelStyle(.titleAndIcon)
                                .padding(.vertical, 4)
                            }
                        }
                    }
                }
                
                ToolbarItem(placement: .primaryAction) {
                    if selectedTab != .comparison {
                        Picker("Engine", selection: $selectedEngineID) {
                            ForEach(engineRegistry.descriptors) { descriptor in
                                Label {
                                    Text(descriptor.displayName)
                                } icon: {
                                    Image(systemName: "shippingbox")
                                }
                                .labelStyle(.titleAndIcon)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                }
            }
        }
    }

    private var selectedEngine: (any LLMEngine)? {
        engineRegistry.engine(for: selectedEngineID)
    }

    private var selectedInferenceSourceName: String {
        return selectedEngine?.descriptor.displayName ?? "Engine"
    }

    private var selectedInferenceCanGenerate: Bool {
        return selectedEngine?.descriptor.capabilities.contains(.inference) ?? false
    }

    private var canRunInferenceComparison: Bool {
        selectedInferenceComparisonEngineIDs.intersection(Set(inferenceAvailableEngines.map(\.id))).isEmpty == false
    }
    private func initializeCheckpointSelection() {
        let availableCheckpointIDs = Set(document.availableCheckpoints.map(\.id))
        if availableCheckpointIDs.contains(selectedCheckpointID) {
            return
        }
        if let currentCheckpointID = document.currentCheckpoint?.id, availableCheckpointIDs.contains(currentCheckpointID) {
            selectedCheckpointID = currentCheckpointID
        } else {
            selectedCheckpointID = document.availableCheckpoints.first?.id ?? ""
        }
    }

    private func startTraining() {
        guard let engine = selectedEngine else {
            return
        }
        guard let trainData = document.trainData,
              let valData = document.valData,
              let checkpointData = document.currentCheckpointData else {
            errorMessage = "Training requires dataset files and a loaded checkpoint."
            return
        }

        errorMessage = nil
        statusMessage = nil
        isRunningTraining = true
        isStoppingTraining = false
        timeToFirstOutputMilliseconds = nil
        trainingTask?.cancel()
        trainingTask = Task {
            do {
                let trainingStart = ContinuousClock.now
                let stream = try await engine.startTraining(
                    request: LLMTrainingRequest(
                        trainData: trainData,
                        validationData: valData,
                        tokenizerData: document.tokenizerData,
                        checkpointData: checkpointData,
                        stepCount: trainingStepCount,
                        batchSize: trainingBatchSize,
                        sequenceLength: trainingSequenceLength,
                        learningRate: trainingLearningRate,
                        validationBatchCount: validationBatchCount
                    )
                )

                var finalProgress = document.trainingProgress
                var isFirstOutput = true
                for try await progress in stream {
                    finalProgress = progress
                    let firstOutputTime: Double?
                    if isFirstOutput {
                        let elapsed = trainingStart.duration(to: .now)
                        firstOutputTime = Double(elapsed.components.seconds) * 1_000
                            + Double(elapsed.components.attoseconds) / 1e15
                        isFirstOutput = false
                    } else {
                        firstOutputTime = nil
                    }
                    await MainActor.run {
                        if let firstOutputTime {
                            timeToFirstOutputMilliseconds = firstOutputTime
                        }
                        document.trainingProgress = progress
                        statusMessage = "Training step \(progress.step) completed."
                    }
                }

                let checkpoint = try await engine.exportCheckpoint()
                await MainActor.run {
                    document.updateCurrentCheckpoint(data: checkpoint, progress: finalProgress)
                    statusMessage = "Training finished. Save the document to persist the current model state."
                    isRunningTraining = false
                    isStoppingTraining = false
                    trainingTask = nil
                }
            } catch is CancellationError {
                await MainActor.run {
                    statusMessage = "Training stopped."
                    isRunningTraining = false
                    isStoppingTraining = false
                    trainingTask = nil
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isRunningTraining = false
                    isStoppingTraining = false
                    trainingTask = nil
                }
            }
        }
    }

    private func stopTraining() {
        guard isRunningTraining, !isStoppingTraining else {
            return
        }
        isStoppingTraining = true
        statusMessage = "Stopping training..."
        trainingTask?.cancel()
    }

    private func runInference(prompt: String, maximumTokenCount: Int, temperature: Double) {
        guard selectedEngine != nil else {
            return
        }
        guard document.currentCheckpointData != nil else {
            errorMessage = "Load the Initial or a manual checkpoint before running inference."
            return
        }

        errorMessage = nil
        statusMessage = nil
        inferenceComparisonErrorMessage = nil
        isRunningInference = true
        inferenceOutput = ""
        inferenceTask?.cancel()

        inferenceTask = Task {
            do {
                let request = LLMInferenceRequest(
                    prompt: prompt,
                    maximumTokenCount: maximumTokenCount,
                    temperature: temperature,
                    tokenizerData: document.tokenizerData,
                    checkpointData: document.currentCheckpointData
                )
                let stream: AsyncThrowingStream<LLMInferenceChunk, Error>
                guard let engine = selectedEngine else {
                    return
                }
                stream = try await engine.startInference(request: request)

                for try await chunk in stream {
                    await MainActor.run {
                        inferenceOutput += chunk.text
                        statusMessage = "Generated \(chunk.generatedTokenCount) tokens."
                    }
                }

                await MainActor.run {
                    isRunningInference = false
                    inferenceTask = nil
                }
            } catch is CancellationError {
                await MainActor.run {
                    statusMessage = inferenceOutput.isEmpty ? "Generation stopped." : "Generation stopped after partial output."
                    isRunningInference = false
                    inferenceTask = nil
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isRunningInference = false
                    inferenceTask = nil
                }
            }
        }
    }

    private func stopInference() {
        inferenceTask?.cancel()
        inferenceTask = nil
    }

    private func startInferenceComparison(prompt: String, maximumTokenCount: Int, temperature: Double) {
        guard !isRunningInference else {
            inferenceComparisonErrorMessage = "Stop text generation before running an inference comparison."
            return
        }
        guard canRunInferenceComparison else {
            inferenceComparisonErrorMessage = "Select at least one inference source."
            return
        }
        guard document.tokenizerData != nil else {
            inferenceComparisonErrorMessage = "Inference comparison requires a tokenizer asset."
            return
        }

        let selectedEngineDescriptors = inferenceAvailableEngines.filter { selectedInferenceComparisonEngineIDs.contains($0.id) }
        let selectedEngines = selectedEngineDescriptors.compactMap { descriptor in
            engineRegistry.engine(for: descriptor.id).map { (descriptor, $0) }
        }
        if !selectedEngines.isEmpty && document.currentCheckpointData == nil {
            inferenceComparisonErrorMessage = "Load the Initial or a manual checkpoint before comparing engine inference."
            return
        }

        inferenceComparisonErrorMessage = nil
        inferenceComparisonStatusMessage = "Preparing inference comparison..."
        inferenceComparisonResults = selectedEngineDescriptors.map {
            InferenceComparisonResult(id: $0.id.rawValue, engineName: $0.displayName, status: .pending, generatedTokenCount: 0)
        }
        isRunningInferenceComparison = true
        inferenceComparisonTask?.cancel()

        let request = LLMInferenceRequest(
            prompt: prompt,
            maximumTokenCount: maximumTokenCount,
            temperature: temperature,
            tokenizerData: document.tokenizerData,
            checkpointData: document.currentCheckpointData
        )

        inferenceComparisonTask = Task {
            var wasCancelled = false
            for (descriptor, engine) in selectedEngines {
                if Task.isCancelled {
                    wasCancelled = true
                    break
                }
                let resultID = descriptor.id.rawValue
                await MainActor.run {
                    if let index = inferenceComparisonResults.firstIndex(where: { $0.id == resultID }) {
                        inferenceComparisonResults[index].status = .running
                    }
                    inferenceComparisonStatusMessage = "Running \(descriptor.displayName)..."
                }

                do {
                    let start = ContinuousClock.now
                    let stream = try await engine.startInference(request: request)
                    var isFirstOutput = true
                    var generatedTokenCount = 0
                    for try await chunk in stream {
                        let elapsed = start.duration(to: .now)
                        let elapsedSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
                        generatedTokenCount = chunk.generatedTokenCount
                        let firstOutputMilliseconds = isFirstOutput ? elapsedSeconds * 1_000 : nil
                        isFirstOutput = false
                        await MainActor.run {
                            if let index = inferenceComparisonResults.firstIndex(where: { $0.id == resultID }) {
                                inferenceComparisonResults[index].generatedTokenCount = chunk.generatedTokenCount
                                inferenceComparisonResults[index].totalElapsedSeconds = elapsedSeconds
                                inferenceComparisonResults[index].tokensPerSecond = elapsedSeconds > 0 ? Double(chunk.generatedTokenCount) / elapsedSeconds : nil
                                if let firstOutputMilliseconds {
                                    inferenceComparisonResults[index].timeToFirstOutputMilliseconds = firstOutputMilliseconds
                                }
                            }
                        }
                    }
                    let totalElapsed = start.duration(to: .now)
                    let totalElapsedSeconds = Double(totalElapsed.components.seconds) + Double(totalElapsed.components.attoseconds) / 1e18
                    await MainActor.run {
                        if let index = inferenceComparisonResults.firstIndex(where: { $0.id == resultID }) {
                            inferenceComparisonResults[index].status = .completed
                            inferenceComparisonResults[index].generatedTokenCount = generatedTokenCount
                            inferenceComparisonResults[index].totalElapsedSeconds = totalElapsedSeconds
                            inferenceComparisonResults[index].tokensPerSecond = totalElapsedSeconds > 0 ? Double(generatedTokenCount) / totalElapsedSeconds : nil
                        }
                    }
                } catch is CancellationError {
                    await MainActor.run {
                        if let index = inferenceComparisonResults.firstIndex(where: { $0.id == resultID }) {
                            inferenceComparisonResults[index].status = .cancelled
                        }
                    }
                    wasCancelled = true
                    break
                } catch {
                    await MainActor.run {
                        if let index = inferenceComparisonResults.firstIndex(where: { $0.id == resultID }) {
                            inferenceComparisonResults[index].status = .failed
                            inferenceComparisonResults[index].errorMessage = error.localizedDescription
                        }
                    }
                }

                await engine.releaseResources()
            }

            await MainActor.run {
                isRunningInferenceComparison = false
                inferenceComparisonTask = nil
                inferenceComparisonStatusMessage = wasCancelled ? "Inference comparison cancelled." : "Inference comparison complete."
            }
        }
    }

    private func stopInferenceComparison() {
        guard isRunningInferenceComparison else {
            return
        }
        inferenceComparisonStatusMessage = "Cancelling inference comparison..."
        inferenceComparisonTask?.cancel()
    }

    private func loadCheckpoint() {
        guard let checkpoint = document.availableCheckpoints.first(where: { $0.id == selectedCheckpointID }) else {
            return
        }
        do {
            try document.loadCheckpoint(checkpoint)
            errorMessage = nil
            timeToFirstOutputMilliseconds = nil
            statusMessage = "Loaded \(checkpoint.name). Training progress has been reset."
            if let checkpointData = document.currentCheckpointData, let engine = selectedEngine {
                Task {
                    try? await engine.loadCheckpoint(data: checkpointData)
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func createManualCheckpoint() {
        document.createManualCheckpoint(named: manualCheckpointName)
        manualCheckpointName = ""
        statusMessage = "Saved a manual checkpoint into the dataset bundle."
        initializeCheckpointSelection()
    }

    private func startComparison() {
        guard let trainData = document.trainData,
              let valData = document.valData,
              let initialCheckpointData = document.initialCheckpointData else {
            comparisonErrorMessage = "Comparison requires dataset files and an initial checkpoint."
            return
        }

        let engines: [any LLMEngine] = comparisonAvailableEngines
            .filter { selectedComparisonEngineIDs.contains($0.id) }
            .compactMap { engineRegistry.engine(for: $0.id) }

        guard !engines.isEmpty else {
            comparisonErrorMessage = "Select at least one training-capable engine."
            return
        }

        comparisonErrorMessage = nil
        comparisonStatusMessage = "Preparing comparison..."
        comparisonResults = engines.map { engine in
            ComparisonEngineResult(
                id: engine.descriptor.id,
                engineName: engine.descriptor.displayName,
                status: .pending,
                progress: .zero
            )
        }
        comparisonLossSamples = []
        isRunningComparison = true

        let tokenizerData = document.tokenizerData
        let batchSize = trainingBatchSize
        let sequenceLength = trainingSequenceLength
        let learningRate = trainingLearningRate
        let validationBatchCount = validationBatchCount
        let stepCount = comparisonStepCount

        comparisonTask?.cancel()
        comparisonTask = Task {
            var wasCancelled = false
            for engine in engines {
                if Task.isCancelled {
                    wasCancelled = true
                    break
                }

                let engineID = engine.descriptor.id
                let engineName = engine.descriptor.displayName

                await MainActor.run {
                    if let index = comparisonResults.firstIndex(where: { $0.id == engineID }) {
                        comparisonResults[index].status = .running
                    }
                    comparisonStatusMessage = "Running \(engineName)..."
                }

                do {
                    try await engine.loadCheckpoint(data: initialCheckpointData)

                    let clockStart = ContinuousClock.now
                    let stream = try await engine.startTraining(
                        request: LLMTrainingRequest(
                            trainData: trainData,
                            validationData: valData,
                            tokenizerData: tokenizerData,
                            checkpointData: initialCheckpointData,
                            stepCount: stepCount,
                            batchSize: batchSize,
                            sequenceLength: sequenceLength,
                            learningRate: learningRate,
                            validationBatchCount: validationBatchCount
                        )
                    )

                    var isFirstOutput = true
                    var finalProgress = LLMTrainingProgress.zero
                    for try await progress in stream {
                        let elapsed = clockStart.duration(to: .now)
                        let elapsedSeconds = Double(elapsed.components.seconds)
                            + Double(elapsed.components.attoseconds) / 1e18
                        finalProgress = progress
                        let firstOutputMilliseconds = isFirstOutput ? elapsedSeconds * 1_000.0 : nil
                        isFirstOutput = false
                        let lossSample = progress.trainingLoss.map {
                            ComparisonLossSample(engineName: engineName, time: elapsedSeconds, loss: $0)
                        }
                        await MainActor.run {
                            if let lossSample {
                                comparisonLossSamples.append(lossSample)
                            }
                            if let index = comparisonResults.firstIndex(where: { $0.id == engineID }) {
                                comparisonResults[index].progress = progress
                                if let firstOutputMilliseconds {
                                    comparisonResults[index].timeToFirstOutputMilliseconds = firstOutputMilliseconds
                                }
                                comparisonResults[index].totalElapsedSeconds = elapsedSeconds
                            }
                        }
                    }

                    await MainActor.run {
                        if let index = comparisonResults.firstIndex(where: { $0.id == engineID }) {
                            comparisonResults[index].progress = finalProgress
                            comparisonResults[index].status = .completed
                        }
                    }
                } catch is CancellationError {
                    await MainActor.run {
                        if let index = comparisonResults.firstIndex(where: { $0.id == engineID }) {
                            comparisonResults[index].status = .cancelled
                        }
                    }
                    wasCancelled = true
                    break
                } catch {
                    let message = error.localizedDescription
                    await MainActor.run {
                        if let index = comparisonResults.firstIndex(where: { $0.id == engineID }) {
                            comparisonResults[index].status = .failed
                            comparisonResults[index].errorMessage = message
                        }
                    }
                }

                // Drop everything the engine still holds (cached checkpoint
                // included) before moving on to the next engine, so an
                // 8-engine comparison doesn't keep ~500MB of trained-checkpoint
                // copy per engine resident after it's done.
                await engine.releaseResources()
            }

            let finalCancelled = wasCancelled
            await MainActor.run {
                isRunningComparison = false
                comparisonTask = nil
                comparisonStatusMessage = finalCancelled ? "Comparison cancelled." : "Comparison complete."
            }
        }
    }

    private func stopComparison() {
        guard isRunningComparison else {
            return
        }
        comparisonStatusMessage = "Cancelling comparison..."
        comparisonTask?.cancel()
    }
}

#Preview("Import Dataset") {
    @Previewable @State var document = LLMDatasetDocument()
    ContentView(document: $document)
}

#Preview("Loaded Dataset", traits: .fixedLayout(width: 800, height: 600)) {
    @Previewable @State var document = LLMDatasetDocument(
        manifest: LLMDatasetManifest(
            tokenizerAssetFileName: "gpt2_tokenizer.bin",
            tokenType: .uint16,
            initialCheckpoint: .init(name: "Initial checkpoint", fileName: "gpt2_124M.bin"),
            trainingProgress: .zero
        ),
        trainData: Data(count: 610520),
        valData: Data(count: 72886),
        tokenizerData: Data(count: 1024),
        initialCheckpointData: Data(count: 1024),
        latestCheckpointData: Data(count: 1024),
        currentCheckpoint: .init(kind: .latest, name: "Latest checkpoint", fileName: "latest-checkpoint.bin"),
        currentCheckpointData: Data(count: 1024)
    )
    ContentView(document: $document)
}
