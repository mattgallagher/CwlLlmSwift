import Foundation

public struct LLMDatasetCheckpointManifestEntry: Codable, Sendable, Equatable, Hashable {
    public var name: String
    public var fileName: String

    public init(name: String, fileName: String) {
        self.name = name
        self.fileName = fileName
    }
}

public struct LLMDatasetManifest: Codable, Sendable, Equatable {
    public static let currentFormatVersion = 3

    public enum TokenType: String, Codable, CaseIterable, Sendable {
        case uint16 = "UInt16"
        case uint32 = "UInt32"

        var byteCount: Int {
            switch self {
            case .uint16:
                2
            case .uint32:
                4
            }
        }
    }

    public var formatVersion: Int
    public var tokenizerAssetFileName: String?
    public var tokenType: TokenType?
    public var initialCheckpoint: LLMDatasetCheckpointManifestEntry?
    public var latestCheckpoint: LLMDatasetCheckpointManifestEntry?
    public var manualCheckpoints: [LLMDatasetCheckpointManifestEntry]
    public var trainingProgress: LLMTrainingProgress

    public init(
        formatVersion: Int = LLMDatasetManifest.currentFormatVersion,
        tokenizerAssetFileName: String? = nil,
        tokenType: TokenType? = nil,
        initialCheckpoint: LLMDatasetCheckpointManifestEntry? = nil,
        latestCheckpoint: LLMDatasetCheckpointManifestEntry? = nil,
        manualCheckpoints: [LLMDatasetCheckpointManifestEntry] = [],
        trainingProgress: LLMTrainingProgress = .zero
    ) {
        self.formatVersion = formatVersion
        self.tokenizerAssetFileName = tokenizerAssetFileName
        self.tokenType = tokenType
        self.initialCheckpoint = initialCheckpoint
        self.latestCheckpoint = latestCheckpoint
        self.manualCheckpoints = manualCheckpoints
        self.trainingProgress = trainingProgress
    }

    enum CodingKeys: String, CodingKey {
        case formatVersion
        case tokenizerAssetFileName
        case tokenType
        case initialCheckpoint
        case latestCheckpoint
        case manualCheckpoints
        case trainingProgress
        case referenceCheckpointName
        case referenceCheckpointFileName
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        formatVersion = try container.decodeIfPresent(Int.self, forKey: .formatVersion) ?? 1
        tokenizerAssetFileName = try container.decodeIfPresent(String.self, forKey: .tokenizerAssetFileName)
        tokenType = try container.decodeIfPresent(TokenType.self, forKey: .tokenType)
        initialCheckpoint = try container.decodeIfPresent(LLMDatasetCheckpointManifestEntry.self, forKey: .initialCheckpoint)
        latestCheckpoint = try container.decodeIfPresent(LLMDatasetCheckpointManifestEntry.self, forKey: .latestCheckpoint)
        manualCheckpoints = try container.decodeIfPresent([LLMDatasetCheckpointManifestEntry].self, forKey: .manualCheckpoints) ?? []
        trainingProgress = try container.decodeIfPresent(LLMTrainingProgress.self, forKey: .trainingProgress) ?? .zero

        if initialCheckpoint == nil,
           let legacyFileName = try container.decodeIfPresent(String.self, forKey: .referenceCheckpointFileName) {
            let legacyName = try container.decodeIfPresent(String.self, forKey: .referenceCheckpointName) ?? "Initial checkpoint"
            initialCheckpoint = LLMDatasetCheckpointManifestEntry(name: legacyName, fileName: legacyFileName)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(formatVersion, forKey: .formatVersion)
        try container.encodeIfPresent(tokenizerAssetFileName, forKey: .tokenizerAssetFileName)
        try container.encodeIfPresent(tokenType, forKey: .tokenType)
        try container.encodeIfPresent(initialCheckpoint, forKey: .initialCheckpoint)
        try container.encodeIfPresent(latestCheckpoint, forKey: .latestCheckpoint)
        try container.encode(manualCheckpoints, forKey: .manualCheckpoints)
        try container.encode(trainingProgress, forKey: .trainingProgress)
    }
}
