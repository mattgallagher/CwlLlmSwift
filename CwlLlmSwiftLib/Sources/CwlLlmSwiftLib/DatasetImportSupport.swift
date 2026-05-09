import Foundation

public enum DatasetFileRole {
    case train
    case validation
    case tokenizer
    case initialCheckpoint

    public var packageFileName: String? {
        switch self {
        case .train:
            return "train.bin"
        case .validation:
            return "val.bin"
        case .tokenizer:
            return nil
        case .initialCheckpoint:
            return nil
        }
    }

    public var expectedFileNameSuffix: String {
        switch self {
        case .train:
            return "train.bin"
        case .validation:
            return "val.bin"
        case .tokenizer:
            return "tokenizer.bin"
        case .initialCheckpoint:
            return ".bin"
        }
    }

    public func matches(fileName: String) -> Bool {
        fileName.localizedLowercase.hasSuffix(expectedFileNameSuffix.localizedLowercase)
    }

    public func inferredName(fromFileName fileName: String) -> String? {
        guard matches(fileName: fileName) else {
            return nil
        }

        let suffixStart = fileName.index(fileName.endIndex, offsetBy: -expectedFileNameSuffix.count)
        let rawPrefix = fileName[..<suffixStart]
            .trimmingCharacters(in: CharacterSet(charactersIn: "_- ."))

        return rawPrefix.isEmpty ? nil : String(rawPrefix)
    }
}

public func validateDatasetFileSelection(url: URL, for role: DatasetFileRole) -> String? {
    guard role.matches(fileName: url.lastPathComponent.localizedLowercase) else {
        return "Selected file must end with \(role.expectedFileNameSuffix)."
    }

    return nil
}
