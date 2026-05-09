import Foundation
import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    static let llmDataset = UTType(exportedAs: "com.cocoawithlove.llmdataset", conformingTo: .package)
}

public enum LLMDatasetValidationIssue: Equatable, Sendable {
    case unsupportedFormatVersion(Int)
    case missingTrainData
    case missingValidationData
    case missingTokenizerAsset(String)
    case missingCheckpoint(String)

    public var message: String {
        switch self {
        case .unsupportedFormatVersion(let version):
            return "Unsupported manifest format version \(version). Expected 1...\(LLMDatasetManifest.currentFormatVersion)."
        case .missingTrainData:
            return "Missing required file \(DatasetFileRole.train.packageFileName ?? DatasetFileRole.train.expectedFileNameSuffix)."
        case .missingValidationData:
            return "Missing required file \(DatasetFileRole.validation.packageFileName ?? DatasetFileRole.validation.expectedFileNameSuffix)."
        case .missingTokenizerAsset(let fileName):
            return "Manifest references missing tokenizer asset \(fileName)."
        case .missingCheckpoint(let fileName):
            return "Manifest references missing checkpoint \(fileName)."
        }
    }
}

enum LLMDatasetDocumentError: LocalizedError, Equatable {
    case incompleteImport
    case emptyBinaryFile(String)
    case invalidTokenAlignment(file: String, tokenType: LLMDatasetManifest.TokenType)
    case duplicateImportFiles
    case duplicateOptionalImportFile(String)
    case missingCheckpointData(String)

    var errorDescription: String? {
        switch self {
        case .incompleteImport:
            return "Choose both dataset binaries and complete the import before saving the document."
        case .emptyBinaryFile(let file):
            return "The selected \(file) file is empty."
        case .invalidTokenAlignment(let file, let tokenType):
            return "The selected \(file) size is not aligned to \(tokenType.rawValue) tokens."
        case .duplicateImportFiles:
            return "\(DatasetFileRole.train.packageFileName ?? DatasetFileRole.train.expectedFileNameSuffix) and \(DatasetFileRole.validation.packageFileName ?? DatasetFileRole.validation.expectedFileNameSuffix) must come from different files."
        case .duplicateOptionalImportFile(let file):
            return "The selected optional asset duplicates another imported file: \(file)."
        case .missingCheckpointData(let fileName):
            return "Checkpoint data for \(fileName) is unavailable."
        }
    }
}

public struct LLMDatasetDocument: FileDocument {
    public static var readableContentTypes: [UTType] { [.llmDataset] }
    public static var writableContentTypes: [UTType] { [.llmDataset] }

    public var manifest: LLMDatasetManifest?
    public var trainData: Data?
    public var valData: Data?
    public var tokenizerData: Data?
    public var initialCheckpointData: Data?
    public var latestCheckpointData: Data?
    public var manualCheckpointData: [String: Data]
    public var currentCheckpoint: LLMCheckpointDescriptor?
    public var currentCheckpointData: Data?
    public var trainingProgress: LLMTrainingProgress
    public private(set) var validationIssues: [LLMDatasetValidationIssue]
    public let engineRegistry: LLMEngineRegistry

    public init(
        manifest: LLMDatasetManifest? = nil,
        trainData: Data? = nil,
        valData: Data? = nil,
        tokenizerData: Data? = nil,
        initialCheckpointData: Data? = nil,
        latestCheckpointData: Data? = nil,
        manualCheckpointData: [String: Data] = [:],
        currentCheckpoint: LLMCheckpointDescriptor? = nil,
        currentCheckpointData: Data? = nil,
        trainingProgress: LLMTrainingProgress? = nil,
        validationIssues: [LLMDatasetValidationIssue]? = nil
    ) {
        self.engineRegistry = LLMEngineRegistry()
        self.manifest = manifest
        self.trainData = trainData
        self.valData = valData
        self.tokenizerData = tokenizerData
        self.initialCheckpointData = initialCheckpointData
        self.latestCheckpointData = latestCheckpointData
        self.manualCheckpointData = manualCheckpointData

        let resolvedCheckpoint = Self.resolveCurrentCheckpoint(
            manifest: manifest,
            currentCheckpoint: currentCheckpoint,
            currentCheckpointData: currentCheckpointData,
            initialCheckpointData: initialCheckpointData,
            latestCheckpointData: latestCheckpointData,
            manualCheckpointData: manualCheckpointData
        )
        self.currentCheckpoint = resolvedCheckpoint.0
        self.currentCheckpointData = resolvedCheckpoint.1
        self.trainingProgress = trainingProgress ?? manifest?.trainingProgress ?? .zero
        self.validationIssues = validationIssues ?? Self.computeValidationIssues(
            manifest: manifest,
            trainData: trainData,
            valData: valData,
            tokenizerData: tokenizerData,
            initialCheckpointData: initialCheckpointData,
            latestCheckpointData: latestCheckpointData,
            manualCheckpointData: manualCheckpointData
        )
    }

    public init(configuration: ReadConfiguration) throws {
        guard configuration.contentType == .llmDataset,
              let rootWrapper = configuration.file.fileWrappers,
              let manifestWrapper = rootWrapper["manifest.json"],
              let manifestData = manifestWrapper.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let manifest = try decoder.decode(LLMDatasetManifest.self, from: manifestData)
        let trainData = rootWrapper[DatasetFileRole.train.packageFileName ?? ""]?.regularFileContents
        let valData = rootWrapper[DatasetFileRole.validation.packageFileName ?? ""]?.regularFileContents
        let tokenizerData = manifest.tokenizerAssetFileName.flatMap { rootWrapper[$0]?.regularFileContents }
        let initialCheckpointData = manifest.initialCheckpoint.flatMap { rootWrapper[$0.fileName]?.regularFileContents }
        let latestCheckpointData = manifest.latestCheckpoint.flatMap { rootWrapper[$0.fileName]?.regularFileContents }
        var manualCheckpointData: [String: Data] = [:]
        for checkpoint in manifest.manualCheckpoints {
            if let data = rootWrapper[checkpoint.fileName]?.regularFileContents {
                manualCheckpointData[checkpoint.fileName] = data
            }
        }

        let resolvedCheckpoint = Self.resolveCurrentCheckpoint(
            manifest: manifest,
            currentCheckpoint: nil,
            currentCheckpointData: nil,
            initialCheckpointData: initialCheckpointData,
            latestCheckpointData: latestCheckpointData,
            manualCheckpointData: manualCheckpointData
        )

        self.engineRegistry = LLMEngineRegistry()
        self.manifest = manifest
        self.trainData = trainData
        self.valData = valData
        self.tokenizerData = tokenizerData
        self.initialCheckpointData = initialCheckpointData
        self.latestCheckpointData = latestCheckpointData
        self.manualCheckpointData = manualCheckpointData
        self.currentCheckpoint = resolvedCheckpoint.0
        self.currentCheckpointData = resolvedCheckpoint.1
        self.trainingProgress = latestCheckpointData == nil ? .zero : manifest.trainingProgress
        self.validationIssues = Self.computeValidationIssues(
            manifest: manifest,
            trainData: trainData,
            valData: valData,
            tokenizerData: tokenizerData,
            initialCheckpointData: initialCheckpointData,
            latestCheckpointData: latestCheckpointData,
            manualCheckpointData: manualCheckpointData
        )
    }

    public func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let manifest = try serializedManifest()
        let fileWrappers = try serializedFileWrappers(manifest: manifest)
        return FileWrapper(directoryWithFileWrappers: fileWrappers)
    }

    func serializedManifest() throws -> LLMDatasetManifest {
        guard var manifest, trainData != nil, valData != nil else {
            throw LLMDatasetDocumentError.incompleteImport
        }

        manifest.formatVersion = LLMDatasetManifest.currentFormatVersion
        manifest.trainingProgress = trainingProgress
        manifest.latestCheckpoint = latestCheckpointData.map {
            _ in LLMDatasetCheckpointManifestEntry(name: "Latest checkpoint", fileName: Self.latestCheckpointFileName)
        }
        return manifest
    }

    func serializedFileWrappers(manifest: LLMDatasetManifest) throws -> [String: FileWrapper] {
        guard let trainData, let valData else {
            throw LLMDatasetDocumentError.incompleteImport
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let manifestWrapper = FileWrapper(regularFileWithContents: try encoder.encode(manifest))
        manifestWrapper.preferredFilename = "manifest.json"

        var fileWrappers: [String: FileWrapper] = [
            "manifest.json": manifestWrapper,
            DatasetFileRole.train.packageFileName ?? "train.bin": FileWrapper(regularFileWithContents: trainData),
            DatasetFileRole.validation.packageFileName ?? "val.bin": FileWrapper(regularFileWithContents: valData),
        ]
        fileWrappers[DatasetFileRole.train.packageFileName ?? "train.bin"]?.preferredFilename = DatasetFileRole.train.packageFileName
        fileWrappers[DatasetFileRole.validation.packageFileName ?? "val.bin"]?.preferredFilename = DatasetFileRole.validation.packageFileName

        if let tokenizerData, let tokenizerFileName = manifest.tokenizerAssetFileName {
            let wrapper = FileWrapper(regularFileWithContents: tokenizerData)
            wrapper.preferredFilename = tokenizerFileName
            fileWrappers[tokenizerFileName] = wrapper
        }

        if let initialCheckpoint = manifest.initialCheckpoint, let initialCheckpointData {
            let wrapper = FileWrapper(regularFileWithContents: initialCheckpointData)
            wrapper.preferredFilename = initialCheckpoint.fileName
            fileWrappers[initialCheckpoint.fileName] = wrapper
        }

        for checkpoint in manifest.manualCheckpoints {
            if let data = manualCheckpointData[checkpoint.fileName] {
                let wrapper = FileWrapper(regularFileWithContents: data)
                wrapper.preferredFilename = checkpoint.fileName
                fileWrappers[checkpoint.fileName] = wrapper
            }
        }

        if let latestCheckpointData {
            let wrapper = FileWrapper(regularFileWithContents: latestCheckpointData)
            wrapper.preferredFilename = Self.latestCheckpointFileName
            fileWrappers[Self.latestCheckpointFileName] = wrapper
        }

        return fileWrappers
    }

    public var isImportingNewDataset: Bool { manifest == nil }
    public var isPackageValid: Bool { manifest != nil && validationIssues.isEmpty }

    public var availableCheckpoints: [LLMCheckpointDescriptor] {
        guard let manifest else {
            return []
        }

        var checkpoints: [LLMCheckpointDescriptor] = []
        if let initialCheckpoint = manifest.initialCheckpoint {
            checkpoints.append(LLMCheckpointDescriptor(kind: .initial, name: initialCheckpoint.name, fileName: initialCheckpoint.fileName))
        }
        checkpoints.append(contentsOf: manifest.manualCheckpoints.map {
            LLMCheckpointDescriptor(kind: .manual, name: $0.name, fileName: $0.fileName)
        })
        return checkpoints
    }

    public mutating func importDataset(
        trainFileURL: URL,
        valFileURL: URL,
        tokenizerFileURL: URL?,
        initialCheckpointURL: URL?,
        manifest: LLMDatasetManifest
    ) throws {
        guard trainFileURL.standardizedFileURL != valFileURL.standardizedFileURL else {
            throw LLMDatasetDocumentError.duplicateImportFiles
        }

        let requiredURLs = [trainFileURL.standardizedFileURL, valFileURL.standardizedFileURL]
        if let tokenizerFileURL {
            let tokenizerURL = tokenizerFileURL.standardizedFileURL
            if requiredURLs.contains(tokenizerURL) || tokenizerURL == initialCheckpointURL?.standardizedFileURL {
                throw LLMDatasetDocumentError.duplicateOptionalImportFile(tokenizerFileURL.lastPathComponent)
            }
        }

        if let initialCheckpointURL, requiredURLs.contains(initialCheckpointURL.standardizedFileURL) {
            throw LLMDatasetDocumentError.duplicateOptionalImportFile(initialCheckpointURL.lastPathComponent)
        }

        let trainData = try Self.readImportedFile(at: trainFileURL)
        let valData = try Self.readImportedFile(at: valFileURL)
        let tokenizerData = try tokenizerFileURL.map(Self.readImportedFile(at:))
        let initialCheckpointData = try initialCheckpointURL.map(Self.readImportedFile(at:))

        try Self.validateImportedData(trainData, fileName: DatasetFileRole.train.packageFileName ?? DatasetFileRole.train.expectedFileNameSuffix, tokenType: manifest.tokenType)
        try Self.validateImportedData(valData, fileName: DatasetFileRole.validation.packageFileName ?? DatasetFileRole.validation.expectedFileNameSuffix, tokenType: manifest.tokenType)
        if let tokenizerData {
            try Self.validateImportedData(tokenizerData, fileName: tokenizerFileURL?.lastPathComponent ?? DatasetFileRole.tokenizer.expectedFileNameSuffix, tokenType: nil)
        }
        if let initialCheckpointData {
            try Self.validateImportedData(initialCheckpointData, fileName: initialCheckpointURL?.lastPathComponent ?? DatasetFileRole.initialCheckpoint.expectedFileNameSuffix, tokenType: nil)
        }

        var resolvedManifest = manifest

        if let tokenizerFileURL {
            resolvedManifest.tokenizerAssetFileName = tokenizerFileURL.lastPathComponent
        } else {
            resolvedManifest.tokenizerAssetFileName = nil
        }

        if let initialCheckpointURL {
            resolvedManifest.initialCheckpoint = LLMDatasetCheckpointManifestEntry(
                name: DatasetFileRole.initialCheckpoint.inferredName(fromFileName: initialCheckpointURL.lastPathComponent) ?? "Initial checkpoint",
                fileName: initialCheckpointURL.lastPathComponent
            )
        } else {
            resolvedManifest.initialCheckpoint = nil
        }
        resolvedManifest.latestCheckpoint = nil
        resolvedManifest.manualCheckpoints = []
        resolvedManifest.trainingProgress = .zero

        self.manifest = resolvedManifest
        self.trainData = trainData
        self.valData = valData
        self.tokenizerData = tokenizerData
        self.initialCheckpointData = initialCheckpointData
        self.latestCheckpointData = initialCheckpointData
        self.manualCheckpointData = [:]
        self.currentCheckpoint = resolvedManifest.initialCheckpoint.map {
            LLMCheckpointDescriptor(kind: .initial, name: $0.name, fileName: $0.fileName)
        }
        self.currentCheckpointData = initialCheckpointData
        self.trainingProgress = .zero
        self.validationIssues = Self.computeValidationIssues(
            manifest: resolvedManifest,
            trainData: trainData,
            valData: valData,
            tokenizerData: tokenizerData,
            initialCheckpointData: initialCheckpointData,
            latestCheckpointData: latestCheckpointData,
            manualCheckpointData: [:]
        )
    }

    public mutating func loadCheckpoint(_ checkpoint: LLMCheckpointDescriptor) throws {
        guard let checkpointData = checkpointData(for: checkpoint) else {
            throw LLMDatasetDocumentError.missingCheckpointData(checkpoint.fileName)
        }
        currentCheckpoint = checkpoint
        currentCheckpointData = checkpointData
        latestCheckpointData = checkpointData
        trainingProgress = .zero
    }

    public mutating func updateCurrentCheckpoint(data: Data, progress: LLMTrainingProgress) {
        currentCheckpointData = data
        latestCheckpointData = data
        currentCheckpoint = LLMCheckpointDescriptor(kind: .latest, name: "Latest checkpoint", fileName: Self.latestCheckpointFileName)
        trainingProgress = progress
    }

    public mutating func createManualCheckpoint(named name: String) {
        guard let currentCheckpointData else {
            return
        }

        let normalizedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let baseName = normalizedName.isEmpty ? "Checkpoint" : normalizedName
        let fileName = Self.uniqueCheckpointFileName(baseName: baseName, existing: Set(manifest?.manualCheckpoints.map(\.fileName) ?? []))
        let entry = LLMDatasetCheckpointManifestEntry(name: baseName, fileName: fileName)
        manifest?.manualCheckpoints.append(entry)
        manualCheckpointData[fileName] = currentCheckpointData
    }

    public func checkpointData(for checkpoint: LLMCheckpointDescriptor) -> Data? {
        switch checkpoint.kind {
        case .initial:
            return initialCheckpointData
        case .latest:
            return latestCheckpointData ?? currentCheckpointData
        case .manual:
            return manualCheckpointData[checkpoint.fileName]
        }
    }

    public static let latestCheckpointFileName = "latest-checkpoint.bin"

    static func computeValidationIssues(
        manifest: LLMDatasetManifest?,
        trainData: Data?,
        valData: Data?,
        tokenizerData: Data?,
        initialCheckpointData: Data?,
        latestCheckpointData: Data?,
        manualCheckpointData: [String: Data]
    ) -> [LLMDatasetValidationIssue] {
        guard let manifest else {
            return []
        }

        var issues: [LLMDatasetValidationIssue] = []
        if !(1...LLMDatasetManifest.currentFormatVersion).contains(manifest.formatVersion) {
            issues.append(.unsupportedFormatVersion(manifest.formatVersion))
        }
        if trainData == nil { issues.append(.missingTrainData) }
        if valData == nil { issues.append(.missingValidationData) }
        if let tokenizerAssetFileName = manifest.tokenizerAssetFileName, tokenizerData == nil {
            issues.append(.missingTokenizerAsset(tokenizerAssetFileName))
        }
        if let initialCheckpoint = manifest.initialCheckpoint, initialCheckpointData == nil {
            issues.append(.missingCheckpoint(initialCheckpoint.fileName))
        }
        if let latestCheckpoint = manifest.latestCheckpoint, latestCheckpointData == nil {
            issues.append(.missingCheckpoint(latestCheckpoint.fileName))
        }
        for checkpoint in manifest.manualCheckpoints where manualCheckpointData[checkpoint.fileName] == nil {
            issues.append(.missingCheckpoint(checkpoint.fileName))
        }
        return issues
    }

    private static func resolveCurrentCheckpoint(
        manifest: LLMDatasetManifest?,
        currentCheckpoint: LLMCheckpointDescriptor?,
        currentCheckpointData: Data?,
        initialCheckpointData: Data?,
        latestCheckpointData: Data?,
        manualCheckpointData: [String: Data]
    ) -> (LLMCheckpointDescriptor?, Data?) {
        if let currentCheckpoint, let currentCheckpointData {
            return (currentCheckpoint, currentCheckpointData)
        }
        if let latestCheckpointData {
            return (
                LLMCheckpointDescriptor(kind: .latest, name: "Latest checkpoint", fileName: Self.latestCheckpointFileName),
                latestCheckpointData
            )
        }
        if let initialCheckpoint = manifest?.initialCheckpoint, let initialCheckpointData {
            return (
                LLMCheckpointDescriptor(kind: .initial, name: initialCheckpoint.name, fileName: initialCheckpoint.fileName),
                initialCheckpointData
            )
        }
        if let firstManual = manifest?.manualCheckpoints.first, let data = manualCheckpointData[firstManual.fileName] {
            return (
                LLMCheckpointDescriptor(kind: .manual, name: firstManual.name, fileName: firstManual.fileName),
                data
            )
        }
        return (nil, nil)
    }

    private static func readImportedFile(at url: URL) throws -> Data {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }
        return try Data(contentsOf: url)
    }

    private static func validateImportedData(
        _ data: Data,
        fileName: String,
        tokenType: LLMDatasetManifest.TokenType?
    ) throws {
        guard !data.isEmpty else {
            throw LLMDatasetDocumentError.emptyBinaryFile(fileName)
        }
        guard let tokenType else {
            return
        }
        guard data.count.isMultiple(of: tokenType.byteCount) else {
            throw LLMDatasetDocumentError.invalidTokenAlignment(file: fileName, tokenType: tokenType)
        }
    }

    private static func uniqueCheckpointFileName(baseName: String, existing: Set<String>) -> String {
        let sanitizedBase = baseName
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        let base = sanitizedBase.isEmpty ? "checkpoint" : sanitizedBase

        var candidate = "\(base).bin"
        var suffix = 2
        while existing.contains(candidate) || candidate == latestCheckpointFileName {
            candidate = "\(base)-\(suffix).bin"
            suffix += 1
        }
        return candidate
    }
}
