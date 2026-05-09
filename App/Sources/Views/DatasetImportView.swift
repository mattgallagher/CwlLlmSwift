import CwlLlmSwiftLib
import SwiftUI
import UniformTypeIdentifiers

struct DatasetImportView: View {
    @Binding var document: LLMDatasetDocument

    @State private var isShowingFileImporter = false
    @State private var importingFileRole: DatasetFileRole?
    @State private var trainFileURL: URL?
    @State private var valFileURL: URL?
    @State private var tokenizerFileURL: URL?
    @State private var initialCheckpointURL: URL?
    @State private var tokenType: LLMDatasetManifest.TokenType? = .uint16
    @State private var errorMessage: String?

    var body: some View {
        TabLayout(title: "Import Dataset", subtitle: "Import training binaries, an optional tokenizer, and an optional initial checkpoint into one `.llmdataset` package.") {
            VStack(alignment: .leading, spacing: 16) {
                fileRow(title: "Training data", selection: trainFileURL, actionTitle: "Choose") { presentImporter(for: .train) }
                fileRow(title: "Validation data", selection: valFileURL, actionTitle: "Choose") { presentImporter(for: .validation) }
                fileRow(title: "Tokenizer", selection: tokenizerFileURL, actionTitle: "Choose") { presentImporter(for: .tokenizer) }
                fileRow(title: "Initial checkpoint", selection: initialCheckpointURL, actionTitle: "Choose") { presentImporter(for: .initialCheckpoint) }

                Picker("Token type", selection: $tokenType) {
                    Text("Not set").tag(LLMDatasetManifest.TokenType?.none)
                    ForEach(LLMDatasetManifest.TokenType.allCases, id: \.self) { type in
                        Text(type.rawValue).tag(Optional(type))
                    }
                }

                if let errorMessage {
                    Text(errorMessage)
                        .foregroundStyle(.red)
                }
            }
        }
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                if trainFileURL == nil || valFileURL == nil {
                    Button("Create", systemImage: "checkmark", action: createDatasetPackage)
                        .disabled(true)
                        .labelStyle(.titleAndIcon)
                } else {
                    Button("Create", systemImage: "checkmark", action: createDatasetPackage)
                        .buttonStyle(.glassProminent)
                        .labelStyle(.titleAndIcon)
                }
            }
        }
        .fileImporter(isPresented: $isShowingFileImporter, allowedContentTypes: [.data], allowsMultipleSelection: false) { result in
            guard let role = importingFileRole else {
                return
            }
            importingFileRole = nil
            selectFile(result, for: role)
        }
    }

    private func presentImporter(for role: DatasetFileRole) {
        importingFileRole = role
        isShowingFileImporter = true
    }

    private func fileRow(title: String, selection: URL?, actionTitle: String, action: @escaping () -> Void) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(selection?.lastPathComponent ?? "Not selected")
                .foregroundStyle(selection == nil ? .secondary : .primary)
            Button(actionTitle, action: action)
        }
    }

    private func selectFile(_ result: Result<[URL], Error>, for role: DatasetFileRole) {
        do {
            guard let url = try result.get().first else {
                return
            }
            if let message = validateDatasetFileSelection(url: url, for: role) {
                errorMessage = message
                return
            }
            switch role {
            case .train:
                trainFileURL = url
            case .validation:
                valFileURL = url
            case .tokenizer:
                tokenizerFileURL = url
            case .initialCheckpoint:
                initialCheckpointURL = url
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func createDatasetPackage() {
        guard let trainFileURL, let valFileURL else {
            errorMessage = "Choose both dataset files before creating the package."
            return
        }

        do {
            try document.importDataset(
                trainFileURL: trainFileURL,
                valFileURL: valFileURL,
                tokenizerFileURL: tokenizerFileURL,
                initialCheckpointURL: initialCheckpointURL,
                manifest: LLMDatasetManifest(tokenType: tokenType)
            )
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
