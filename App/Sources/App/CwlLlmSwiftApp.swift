import CwlLlmSwiftLib
import SwiftUI

@main
struct CwlLlmSwiftApp: App {
    var body: some Scene {
        DocumentGroup(newDocument: LLMDatasetDocument()) { file in
            ContentView(document: file.$document).id("document")
        }
    }
}
