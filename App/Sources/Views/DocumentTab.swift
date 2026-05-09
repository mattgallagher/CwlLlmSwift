import Foundation

enum DocumentTab: String, CaseIterable, Hashable, Identifiable {
    case training
    case inference
    case comparison

    var id: Self { self }

    var title: String {
        switch self {
        case .training: "Training"
        case .inference: "Inference"
        case .comparison: "Comparison"
        }
    }

    var systemImage: String {
        switch self {
        case .training: "chart.line.uptrend.xyaxis"
        case .inference: "text.quote"
        case .comparison: "chart.bar.xaxis"
        }
    }
}
