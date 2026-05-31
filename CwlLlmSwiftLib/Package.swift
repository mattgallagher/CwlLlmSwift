// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "CwlLlmSwiftLib",
    platforms: [
        .macOS(.v26),
    ],
    products: [
        .library(
            name: "CwlLlmSwiftLib",
            targets: ["CwlLlmSwiftLib"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics.git", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "CLLMAMX",
            path: "Sources/CLLMAMX",
            publicHeadersPath: "export",
            cSettings: [
                .unsafeFlags([
                    "-O3",
                    "-ffast-math",
                    "-Wno-shorten-64-to-32",
                    "-Wno-gnu-folding-constant",
                ]),
            ]
        ),
        .target(
            name: "CLLMCReference",
            path: "Sources/CLLMCReference",
            publicHeadersPath: "export",
            cSettings: [
                .define("TESTING"),
                .headerSearchPath("include"),
                .unsafeFlags([
                    "-O3",
                    "-ffast-math",
                    "-Wno-deprecated",
                    "-Wno-shorten-64-to-32",
                    "-Wno-gnu-folding-constant",
                ]),
            ]
        ),
        .target(
            name: "CwlLlmSwiftLib",
            dependencies: [
                "CLLMAMX",
                "CLLMCReference",
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            path: "Sources/CwlLlmSwiftLib",
            swiftSettings: [
                .unsafeFlags([
                    "-remove-runtime-asserts",
                    "-Xcc",
                    "-DACCELERATE_NEW_LAPACK",
                ]),
            ]
        ),
        .testTarget(
            name: "CwlLlmSwiftLibTests",
            dependencies: [
                "CwlLlmSwiftLib",
                "CLLMCReference",
            ],
            path: "Tests/CwlLlmSwiftLibTests"
        ),
    ]
)
