import XCTest
@testable import CwlLlmSwiftLib

final class LLMDatasetManifestTests: XCTestCase {
    func testManifestRoundTripPreservesFields() throws {
        let manifest = LLMDatasetManifest(
            tokenizerAssetFileName: "gpt2_tokenizer.bin",
            tokenType: .uint16,
            initialCheckpoint: .init(name: "gpt2_124M", fileName: "gpt2_124M.bin"),
            manualCheckpoints: [.init(name: "epoch-1", fileName: "epoch-1.bin")],
            trainingProgress: LLMTrainingProgress(
                step: 4,
                iterationsPerSecond: 2,
                forwardPassMilliseconds: 1.25,
                backwardPassMilliseconds: 2.5,
                trainingLoss: 0.75,
                validationLoss: 0.8
            )
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let data = try encoder.encode(manifest)
        let decoded = try decoder.decode(LLMDatasetManifest.self, from: data)

        XCTAssertEqual(decoded, manifest)
    }

    func testTrainingProgressDecodesWhenPassTimingsAreMissing() throws {
        let progressData = try JSONSerialization.data(withJSONObject: [
            "step": 4,
            "iterationsPerSecond": 2.0,
            "trainingLoss": 0.75,
            "validationLoss": 0.8,
        ])

        let progress = try JSONDecoder().decode(LLMTrainingProgress.self, from: progressData)

        XCTAssertEqual(progress.step, 4)
        XCTAssertEqual(progress.iterationsPerSecond, 2.0)
        XCTAssertNil(progress.forwardPassMilliseconds)
        XCTAssertNil(progress.backwardPassMilliseconds)
        XCTAssertEqual(progress.trainingLoss, 0.75)
        XCTAssertEqual(progress.validationLoss, 0.8)
    }

    func testDocumentValidationFlagsMissingBinaryFiles() {
        let document = LLMDatasetDocument()

        XCTAssertNil(document.manifest)
        XCTAssertTrue(document.validationIssues.isEmpty)
        XCTAssertFalse(document.isPackageValid)
        XCTAssertTrue(document.isImportingNewDataset)
    }

    func testDocumentValidationAcceptsCompleteCurrentFormatPackage() {
        let document = LLMDatasetDocument(
            manifest: LLMDatasetManifest(
                tokenizerAssetFileName: "gpt2_tokenizer.bin",
                tokenType: .uint16,
                initialCheckpoint: .init(name: "gpt2_124M", fileName: "gpt2_124M.bin")
            ),
            trainData: Data([0x00, 0x01]),
            valData: Data([0x02, 0x03]),
            tokenizerData: Data([0x04]),
            initialCheckpointData: Data([0x05])
        )

        XCTAssertTrue(document.validationIssues.isEmpty)
        XCTAssertTrue(document.isPackageValid)
    }

    func testDocumentValidationFlagsUnsupportedManifestVersion() {
        let document = LLMDatasetDocument(
            manifest: LLMDatasetManifest(formatVersion: 99),
            trainData: Data([0x00]),
            valData: Data([0x01])
        )

        XCTAssertEqual(document.validationIssues, [.unsupportedFormatVersion(99)])
    }

    func testImportDatasetInfersTokenCountsFromSelectedTokenType() throws {
        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: temporaryDirectory)
        }

        let trainURL = temporaryDirectory.appendingPathComponent("train.bin")
        let valURL = temporaryDirectory.appendingPathComponent("val.bin")

        try Data([0x00, 0x01, 0x02, 0x03]).write(to: trainURL)
        try Data([0x04, 0x05]).write(to: valURL)

        var document = LLMDatasetDocument()
        try document.importDataset(
            trainFileURL: trainURL,
            valFileURL: valURL,
            tokenizerFileURL: nil,
            initialCheckpointURL: nil,
            manifest: LLMDatasetManifest(tokenType: .uint16)
        )

        XCTAssertEqual(document.manifest?.tokenType, .uint16)
        XCTAssertTrue(document.isPackageValid)
    }

    func testDatasetFileSelectionRejectsWrongTrainFileName() {
        let url = URL(fileURLWithPath: "/tmp/val.bin")

        XCTAssertEqual(
            validateDatasetFileSelection(url: url, for: .train),
            "Selected file must end with train.bin."
        )
    }

    func testDatasetFileSelectionAcceptsExpectedValidationFileName() {
        let url = URL(fileURLWithPath: "/tmp/tiny_shakespeare_val.bin")

        XCTAssertNil(validateDatasetFileSelection(url: url, for: .validation))
    }

    func testDatasetFileSelectionAcceptsExpectedTrainFileSuffix() {
        let url = URL(fileURLWithPath: "/tmp/tiny_shakespeare_train.bin")

        XCTAssertNil(validateDatasetFileSelection(url: url, for: .train))
    }

    func testDocumentValidationFlagsMissingReferencedOptionalAssets() {
        let document = LLMDatasetDocument(
            manifest: LLMDatasetManifest(
                tokenizerAssetFileName: "gpt2_tokenizer.bin",
                initialCheckpoint: .init(name: "gpt2_124M", fileName: "gpt2_124M.bin")
            ),
            trainData: Data([0x00]),
            valData: Data([0x01])
        )

        XCTAssertEqual(
            document.validationIssues,
            [
                .missingTokenizerAsset("gpt2_tokenizer.bin"),
                .missingCheckpoint("gpt2_124M.bin"),
            ]
        )
    }

    func testImportDatasetPersistsOptionalTokenizerAndReferenceCheckpointAssets() throws {
        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: temporaryDirectory)
        }

        let trainURL = temporaryDirectory.appendingPathComponent("tiny_shakespeare_train.bin")
        let valURL = temporaryDirectory.appendingPathComponent("tiny_shakespeare_val.bin")
        let tokenizerURL = temporaryDirectory.appendingPathComponent("gpt2_tokenizer.bin")
        let checkpointURL = temporaryDirectory.appendingPathComponent("gpt2_124M.bin")

        try Data([0x00, 0x01]).write(to: trainURL)
        try Data([0x02, 0x03]).write(to: valURL)
        try Data([0x04, 0x05, 0x06]).write(to: tokenizerURL)
        try Data([0x07, 0x08, 0x09, 0x0A]).write(to: checkpointURL)

        var document = LLMDatasetDocument()
        try document.importDataset(
            trainFileURL: trainURL,
            valFileURL: valURL,
            tokenizerFileURL: tokenizerURL,
            initialCheckpointURL: checkpointURL,
            manifest: LLMDatasetManifest()
        )

        XCTAssertEqual(document.manifest?.tokenizerAssetFileName, "gpt2_tokenizer.bin")
        XCTAssertEqual(document.manifest?.initialCheckpoint?.fileName, "gpt2_124M.bin")
        XCTAssertEqual(document.manifest?.initialCheckpoint?.name, "gpt2_124M")
        XCTAssertEqual(document.tokenizerData, Data([0x04, 0x05, 0x06]))
        XCTAssertEqual(document.initialCheckpointData, Data([0x07, 0x08, 0x09, 0x0A]))
    }

    func testDatasetFileSelectionAcceptsTokenizerSuffix() {
        let url = URL(fileURLWithPath: "/tmp/gpt2_tokenizer.bin")

        XCTAssertNil(validateDatasetFileSelection(url: url, for: .tokenizer))
    }

    func testDatasetFileSelectionAcceptsInitialCheckpointBinSuffix() {
        let url = URL(fileURLWithPath: "/tmp/gpt2_124M.bin")

        XCTAssertNil(validateDatasetFileSelection(url: url, for: .initialCheckpoint))
    }

    func testLoadingCheckpointResetsPersistedProgress() throws {
        var document = LLMDatasetDocument(
            manifest: LLMDatasetManifest(
                initialCheckpoint: .init(name: "Initial checkpoint", fileName: "initial.bin"),
                manualCheckpoints: [.init(name: "Epoch 1", fileName: "epoch-1.bin")],
                trainingProgress: LLMTrainingProgress(step: 12, iterationsPerSecond: 3.5, trainingLoss: 1.25, validationLoss: 1.5)
            ),
            trainData: Data([0x00]),
            valData: Data([0x01]),
            initialCheckpointData: Data([0x10]),
            latestCheckpointData: Data([0x20]),
            manualCheckpointData: ["epoch-1.bin": Data([0x30])],
            currentCheckpoint: .init(kind: .latest, name: "Latest checkpoint", fileName: "latest-checkpoint.bin"),
            currentCheckpointData: Data([0x20]),
            trainingProgress: LLMTrainingProgress(step: 12, iterationsPerSecond: 3.5, trainingLoss: 1.25, validationLoss: 1.5)
        )

        try document.loadCheckpoint(.init(kind: .manual, name: "Epoch 1", fileName: "epoch-1.bin"))

        XCTAssertEqual(document.currentCheckpoint?.kind, .manual)
        XCTAssertEqual(document.currentCheckpointData, Data([0x30]))
        XCTAssertEqual(document.trainingProgress, .zero)
    }

    func testUpdatingCurrentCheckpointPromotesLatestCheckpointAndProgress() {
        var document = LLMDatasetDocument(
            manifest: LLMDatasetManifest(),
            trainData: Data([0x00]),
            valData: Data([0x01])
        )
        let progress = LLMTrainingProgress(step: 4, iterationsPerSecond: 2, trainingLoss: 0.75, validationLoss: 0.8)

        document.updateCurrentCheckpoint(data: Data([0xAB, 0xCD]), progress: progress)

        XCTAssertEqual(document.currentCheckpoint?.kind, .latest)
        XCTAssertEqual(document.latestCheckpointData, Data([0xAB, 0xCD]))
        XCTAssertEqual(document.trainingProgress, progress)
    }

    func testAvailableCheckpointsExcludeImplicitLatestCheckpoint() {
        let document = LLMDatasetDocument(
            manifest: LLMDatasetManifest(
                initialCheckpoint: .init(name: "Initial checkpoint", fileName: "initial.bin"),
                manualCheckpoints: [.init(name: "Epoch 1", fileName: "epoch-1.bin")]
            ),
            trainData: Data([0x00]),
            valData: Data([0x01]),
            initialCheckpointData: Data([0x10]),
            latestCheckpointData: Data([0x20]),
            manualCheckpointData: ["epoch-1.bin": Data([0x30])],
            currentCheckpoint: .init(kind: .latest, name: "Latest checkpoint", fileName: "latest-checkpoint.bin"),
            currentCheckpointData: Data([0x20])
        )

        XCTAssertEqual(document.availableCheckpoints.map(\.kind), [.initial, .manual])
    }

    func testSavingInitialCheckpointOnlyDocumentDoesNotInventLatestCheckpoint() throws {
        let document = LLMDatasetDocument(
            manifest: LLMDatasetManifest(
                initialCheckpoint: .init(name: "Initial checkpoint", fileName: "initial.bin")
            ),
            trainData: Data([0x00]),
            valData: Data([0x01]),
            initialCheckpointData: Data([0x10]),
            currentCheckpoint: .init(kind: .initial, name: "Initial checkpoint", fileName: "initial.bin"),
            currentCheckpointData: Data([0x10])
        )

        let manifest = try document.serializedManifest()
        let wrappers = try document.serializedFileWrappers(manifest: manifest)
        XCTAssertNil(wrappers[LLMDatasetDocument.latestCheckpointFileName])

        let manifestData = try XCTUnwrap(wrappers["manifest.json"]?.regularFileContents)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let savedManifest = try decoder.decode(LLMDatasetManifest.self, from: manifestData)
        XCTAssertNil(savedManifest.latestCheckpoint)
    }

    func testManifestDecodesLegacyReferenceCheckpointFields() throws {
        let manifestData = try JSONSerialization.data(withJSONObject: [
            "formatVersion": 1,
            "referenceCheckpointName": "gpt2_124M",
            "referenceCheckpointFileName": "gpt2_124M.bin",
        ])

        let manifest = try JSONDecoder().decode(LLMDatasetManifest.self, from: manifestData)

        XCTAssertEqual(manifest.initialCheckpoint, .init(name: "gpt2_124M", fileName: "gpt2_124M.bin"))
    }
}
