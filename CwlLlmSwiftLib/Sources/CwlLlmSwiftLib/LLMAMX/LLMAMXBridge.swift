import CLLMAMX
import Dispatch
import Foundation
import simd

private struct SendableUnsafeMutableBuffer<Element>: @unchecked Sendable {
    let baseAddress: UnsafeMutablePointer<Element>
}

enum LLMAMXBridge {
    private static let tileRows = 16
    private static let tileColumns = 64
    private static let tileColumnsPerAccumulator = 16
    private static let accumulatorCount = 4
    #if os(macOS) && arch(arm64)
    private static let amxMatFPF32: UInt64 = 4 << 42
    private static let zeroTileRow = [Float](repeating: 0, count: tileRows)
    #endif

    static var isAvailable: Bool {
        #if os(macOS) && arch(arm64)
        true
        #else
        false
        #endif
    }

    static func gemm(
        out: inout [Float],
        outRowStride: Int,
        lhs: [Float],
        lhsRowStride: Int,
        lhsColStride: Int,
        rhs: [Float],
        rhsRowStride: Int,
        rhsColStride: Int,
        rowCount: Int,
        columnCount: Int,
        innerCount: Int,
        accumulate: Bool
    ) {
        precondition(outRowStride >= columnCount)
        guard rowCount > 0, columnCount > 0, innerCount > 0 else {
            return
        }

        if !isAvailable {
            scalarGemm(
                out: &out,
                outRowStride: outRowStride,
                lhs: lhs,
                lhsRowStride: lhsRowStride,
                lhsColStride: lhsColStride,
                rhs: rhs,
                rhsRowStride: rhsRowStride,
                rhsColStride: rhsColStride,
                rowCount: rowCount,
                columnCount: columnCount,
                innerCount: innerCount,
                accumulate: accumulate
            )
            return
        }

        #if os(macOS) && arch(arm64)

        let columnBlockCount = (columnCount + tileColumns - 1) / tileColumns
        let workerCount = max(1, min(ProcessInfo.processInfo.activeProcessorCount, columnBlockCount))
        let lhs = lhs.span
        let rhs = rhs.span

        out.withUnsafeMutableBufferPointer { outBuffer in
            guard let outBase = outBuffer.baseAddress else {
                return
            }

            let outStorage = SendableUnsafeMutableBuffer(baseAddress: outBase)

            DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
                var lhsPanel = Array(repeating: Float.zero, count: innerCount * tileRows)
                var rhsPanels = Array(repeating: Float.zero, count: innerCount * tileColumns)
                var outTiles = Array(repeating: Float.zero, count: tileRows * tileColumns)

                lhsPanel.withUnsafeMutableBufferPointer { lhsPanelBuffer in
                    rhsPanels.withUnsafeMutableBufferPointer { rhsPanelsBuffer in
                        outTiles.withUnsafeMutableBufferPointer { outTilesBuffer in
                            guard let lhsPanelBase = lhsPanelBuffer.baseAddress,
                                  let rhsPanelsBase = rhsPanelsBuffer.baseAddress,
                                  let outTilesBase = outTilesBuffer.baseAddress else {
                                return
                            }

                            amx_set()
                            defer { amx_clr() }

                            for columnBlock in stride(from: worker, to: columnBlockCount, by: workerCount) {
                                let columnStart = columnBlock * tileColumns
                                let columnsInBlock = min(tileColumns, columnCount - columnStart)
                                packRHS(
                                    storage: rhs,
                                    columnStart: columnStart,
                                    columnsInBlock: columnsInBlock,
                                    innerCount: innerCount,
                                    rowStride: rhsRowStride,
                                    colStride: rhsColStride,
                                    dst: rhsPanelsBase
                                )

                                for rowStart in stride(from: 0, to: rowCount, by: tileRows) {
                                    let rowsInBlock = min(tileRows, rowCount - rowStart)
                                    packLHS(
                                        storage: lhs,
                                        rowStart: rowStart,
                                        rowsInBlock: rowsInBlock,
                                        innerCount: innerCount,
                                        rowStride: lhsRowStride,
                                        colStride: lhsColStride,
                                        dst: lhsPanelBase
                                    )

                                    amxF32_16x64(
                                        outTiles: outTilesBase,
                                        lhsPanel: lhsPanelBase,
                                        rhsPanels: rhsPanelsBase,
                                        innerCount: innerCount
                                    )

                                    scatterOutput(
                                        outTiles: outTilesBase,
                                        outBase: outStorage.baseAddress,
                                        outRowStride: outRowStride,
                                        rowStart: rowStart,
                                        rowsInBlock: rowsInBlock,
                                        columnStart: columnStart,
                                        columnsInBlock: columnsInBlock,
                                        accumulate: accumulate
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        #else
        scalarGemm(
            out: &out,
            outRowStride: outRowStride,
            lhs: lhs,
            lhsRowStride: lhsRowStride,
            lhsColStride: lhsColStride,
            rhs: rhs,
            rhsRowStride: rhsRowStride,
            rhsColStride: rhsColStride,
            rowCount: rowCount,
            columnCount: columnCount,
            innerCount: innerCount,
            accumulate: accumulate
        )
        #endif
    }

    private static func scalarGemm(
        out: inout [Float],
        outRowStride: Int,
        lhs: [Float],
        lhsRowStride: Int,
        lhsColStride: Int,
        rhs: [Float],
        rhsRowStride: Int,
        rhsColStride: Int,
        rowCount: Int,
        columnCount: Int,
        innerCount: Int,
        accumulate: Bool
    ) {
        for row in 0..<rowCount {
            for column in 0..<columnCount {
                var value = accumulate ? out[row * outRowStride + column] : 0
                for k in 0..<innerCount {
                    value += lhs[row * lhsRowStride + k * lhsColStride] * rhs[column * rhsRowStride + k * rhsColStride]
                }
                out[row * outRowStride + column] = value
            }
        }
    }

    #if os(macOS) && arch(arm64)
    private static func amxF32_16x64(
        outTiles: UnsafeMutablePointer<Float>,
        lhsPanel: UnsafePointer<Float>,
        rhsPanels: UnsafePointer<Float>,
        innerCount: Int
    ) {
        zeroTileRow.withUnsafeBufferPointer { zeroBuffer in
            guard let zeroBase = zeroBuffer.baseAddress else {
                return
            }

            for tile in 0..<accumulatorCount {
                for row in 0..<tileRows {
                    amx_ldz(zeroBase.amxZOperand(row: UInt32(tile + (row * accumulatorCount))))
                }
            }

            for k in 0..<innerCount {
                let lhsBase = lhsPanel + (k * tileRows)
                amx_ldx(lhsBase.amxXYOperand)
                for tile in 0..<accumulatorCount {
                    let rhsBase = rhsPanels + (tile * innerCount * tileRows) + (k * tileRows)
                    amx_ldy(rhsBase.amxXYOperand)
                    amx_matfp(amxMatFPF32 | (UInt64(tile) << 20))
                }
            }

            for tile in 0..<accumulatorCount {
                let tileBase = outTiles + (tile * tileRows * tileRows)
                for row in 0..<tileRows {
                    let rowBase = UnsafePointer<Float>(tileBase + (row * tileRows))
                    amx_stz(rowBase.amxZOperand(row: UInt32(tile + (row * accumulatorCount))))
                }
            }
        }
    }
    #endif

    private static func packLHS(
        storage: Span<Float>,
        rowStart: Int,
        rowsInBlock: Int,
        innerCount: Int,
        rowStride: Int,
        colStride: Int,
        dst: UnsafeMutablePointer<Float>
    ) {
        let dstRaw = UnsafeMutableRawPointer(dst)

        for k in 0..<innerCount {
            let dstBase = k * tileRows
            let srcBase = rowStart * rowStride + k * colStride

            if rowsInBlock == tileRows {
                dstRaw.storeBytes(
                    of: SIMD4<Float>(
                        storage[srcBase],
                        storage[srcBase + rowStride],
                        storage[srcBase + rowStride * 2],
                        storage[srcBase + rowStride * 3]
                    ),
                    toByteOffset: (dstBase + 0) * MemoryLayout<Float>.stride,
                    as: SIMD4<Float>.self
                )
                dstRaw.storeBytes(
                    of: SIMD4<Float>(
                        storage[srcBase + rowStride * 4],
                        storage[srcBase + rowStride * 5],
                        storage[srcBase + rowStride * 6],
                        storage[srcBase + rowStride * 7]
                    ),
                    toByteOffset: (dstBase + 4) * MemoryLayout<Float>.stride,
                    as: SIMD4<Float>.self
                )
                dstRaw.storeBytes(
                    of: SIMD4<Float>(
                        storage[srcBase + rowStride * 8],
                        storage[srcBase + rowStride * 9],
                        storage[srcBase + rowStride * 10],
                        storage[srcBase + rowStride * 11]
                    ),
                    toByteOffset: (dstBase + 8) * MemoryLayout<Float>.stride,
                    as: SIMD4<Float>.self
                )
                dstRaw.storeBytes(
                    of: SIMD4<Float>(
                        storage[srcBase + rowStride * 12],
                        storage[srcBase + rowStride * 13],
                        storage[srcBase + rowStride * 14],
                        storage[srcBase + rowStride * 15]
                    ),
                    toByteOffset: (dstBase + 12) * MemoryLayout<Float>.stride,
                    as: SIMD4<Float>.self
                )
                continue
            }

            var row = 0
            while row < rowsInBlock {
                dst[dstBase + row] = storage[srcBase + row * rowStride]
                row += 1
            }
            while row < tileRows {
                dst[dstBase + row] = 0
                row += 1
            }
        }
    }

    private static func packRHS(
        storage: Span<Float>,
        columnStart: Int,
        columnsInBlock: Int,
        innerCount: Int,
        rowStride: Int,
        colStride: Int,
        dst: UnsafeMutablePointer<Float>
    ) {
        let dstRaw = UnsafeMutableRawPointer(dst)

        for tile in 0..<accumulatorCount {
            let tileColumnStart = tile * tileRows
            for k in 0..<innerCount {
                let dstBase = tile * innerCount * tileRows + k * tileRows
                let srcBase = columnStart * rowStride + k * colStride

                if rowStride == 1 && tileColumnStart + tileRows <= columnsInBlock {
                    dstRaw.storeBytes(
                        of: SIMD4<Float>(
                            storage[srcBase + tileColumnStart],
                            storage[srcBase + tileColumnStart + 1],
                            storage[srcBase + tileColumnStart + 2],
                            storage[srcBase + tileColumnStart + 3]
                        ),
                        toByteOffset: (dstBase + 0) * MemoryLayout<Float>.stride,
                        as: SIMD4<Float>.self
                    )
                    dstRaw.storeBytes(
                        of: SIMD4<Float>(
                            storage[srcBase + tileColumnStart + 4],
                            storage[srcBase + tileColumnStart + 5],
                            storage[srcBase + tileColumnStart + 6],
                            storage[srcBase + tileColumnStart + 7]
                        ),
                        toByteOffset: (dstBase + 4) * MemoryLayout<Float>.stride,
                        as: SIMD4<Float>.self
                    )
                    dstRaw.storeBytes(
                        of: SIMD4<Float>(
                            storage[srcBase + tileColumnStart + 8],
                            storage[srcBase + tileColumnStart + 9],
                            storage[srcBase + tileColumnStart + 10],
                            storage[srcBase + tileColumnStart + 11]
                        ),
                        toByteOffset: (dstBase + 8) * MemoryLayout<Float>.stride,
                        as: SIMD4<Float>.self
                    )
                    dstRaw.storeBytes(
                        of: SIMD4<Float>(
                            storage[srcBase + tileColumnStart + 12],
                            storage[srcBase + tileColumnStart + 13],
                            storage[srcBase + tileColumnStart + 14],
                            storage[srcBase + tileColumnStart + 15]
                        ),
                        toByteOffset: (dstBase + 12) * MemoryLayout<Float>.stride,
                        as: SIMD4<Float>.self
                    )
                    continue
                }

                for lane in 0..<tileRows {
                    let column = tileColumnStart + lane
                    dst[dstBase + lane] = column < columnsInBlock ? storage[srcBase + column * rowStride] : 0
                }
            }
        }
    }

    private static func scatterOutput(
        outTiles: UnsafeMutablePointer<Float>,
        outBase: UnsafeMutablePointer<Float>,
        outRowStride: Int,
        rowStart: Int,
        rowsInBlock: Int,
        columnStart: Int,
        columnsInBlock: Int,
        accumulate: Bool
    ) {
        let fullTileCount = columnsInBlock / tileRows
        let tailColumns = columnsInBlock % tileRows

        for row in 0..<rowsInBlock {
            var dst = UnsafeMutableRawPointer(outBase + ((rowStart + row) * outRowStride) + columnStart)

            for tile in 0..<fullTileCount {
                let tileBase = outTiles + (tile * tileRows * tileRows) + row
                let v0 = SIMD4<Float>(tileBase[0], tileBase[16], tileBase[32], tileBase[48])
                let v1 = SIMD4<Float>(tileBase[64], tileBase[80], tileBase[96], tileBase[112])
                let v2 = SIMD4<Float>(tileBase[128], tileBase[144], tileBase[160], tileBase[176])
                let v3 = SIMD4<Float>(tileBase[192], tileBase[208], tileBase[224], tileBase[240])

                if accumulate {
                    dst.storeBytes(
                        of: dst.load(fromByteOffset: 0, as: SIMD4<Float>.self) + v0,
                        toByteOffset: 0,
                        as: SIMD4<Float>.self
                    )
                    dst.storeBytes(
                        of: dst.load(fromByteOffset: 16, as: SIMD4<Float>.self) + v1,
                        toByteOffset: 16,
                        as: SIMD4<Float>.self
                    )
                    dst.storeBytes(
                        of: dst.load(fromByteOffset: 32, as: SIMD4<Float>.self) + v2,
                        toByteOffset: 32,
                        as: SIMD4<Float>.self
                    )
                    dst.storeBytes(
                        of: dst.load(fromByteOffset: 48, as: SIMD4<Float>.self) + v3,
                        toByteOffset: 48,
                        as: SIMD4<Float>.self
                    )
                } else {
                    dst.storeBytes(of: v0, toByteOffset: 0, as: SIMD4<Float>.self)
                    dst.storeBytes(of: v1, toByteOffset: 16, as: SIMD4<Float>.self)
                    dst.storeBytes(of: v2, toByteOffset: 32, as: SIMD4<Float>.self)
                    dst.storeBytes(of: v3, toByteOffset: 48, as: SIMD4<Float>.self)
                }

                dst += tileRows * MemoryLayout<Float>.stride
            }

            if tailColumns > 0 {
                let tileBase = outTiles + (fullTileCount * tileRows * tileRows) + row
                let dstTyped = dst.assumingMemoryBound(to: Float.self)
                for column in 0..<tailColumns {
                    let value = tileBase[column * tileRows]
                    if accumulate {
                        dstTyped[column] += value
                    } else {
                        dstTyped[column] = value
                    }
                }
            }
        }
    }
}

private extension UnsafePointer<Float> {
    var amxXYOperand: UInt64 {
        UInt64(UInt(Int(bitPattern: UnsafeRawPointer(self)))) & 0x00FF_FFFF_FFFF_FFFF
    }
    
    func amxZOperand(row: UInt32) -> UInt64 {
        (UInt64(UInt(Int(bitPattern: UnsafeRawPointer(self)))) & 0x00FF_FFFF_FFFF_FFFF) | (UInt64(row & 63) << 56)
    }
}
