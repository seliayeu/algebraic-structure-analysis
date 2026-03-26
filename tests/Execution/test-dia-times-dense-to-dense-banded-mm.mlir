// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner - --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

module {
  func.func private @printMemrefF32(memref<*xf32>)
  
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index

    // =========================================================================
    // Variation 1: Main Diagonal Only (Lower=0, Upper=0)
    // =========================================================================
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK-NEXT: {{\[}}[5{{.*}}, 5{{.*}}, 0{{.*}}],
    // CHECK-NEXT:  [2{{.*}}, 4{{.*}}, 4{{.*}}],
    // CHECK-NEXT:  [0{{.*}}, 6{{.*}}, 9{{.*}}]]

    %r0 = tensor.empty(%c3) : tensor<?x3xf32>
    %dia = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[5.0, 2.0, 3.0]]> : tensor<1x3xf32>
    %dense = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 1.0],
               [1.0, 2.0, 2.0],
               [1.0, 2.0, 3.0]]> : tensor<3x3xf32>
    %1 = dia.matmul ins(%dia, %dense: tensor<1x3xf32>, tensor<3x3xf32>)
                    outs(%r0 : tensor<?x3xf32>) -> tensor<?x3xf32>

    %dim = tensor.dim %1, %c0 : tensor<?x3xf32>
    %memref = memref.alloc(%dim) : memref<?x3xf32>
    bufferization.materialize_in_destination %1 in writable %memref
        : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast = memref.cast %memref : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<?x3xf32>

    // =========================================================================
    // Variation 2: Lower Bandwidth Present (Lower=1, Upper=0)
    // =========================================================================
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK-NEXT: {{\[}}[2{{.*}}, 2{{.*}}, 2{{.*}}],
    // CHECK-NEXT:  [7{{.*}}, 7{{.*}}, 4{{.*}}],
    // CHECK-NEXT:  [8{{.*}}, 8{{.*}}, 10{{.*}}]]

    %r1 = tensor.empty(%c3) : tensor<?x3xf32>
    %dia2 = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 1.0, 2.0],
               [2.0, 3.0, 4.0]]> : tensor<2x3xf32>
    %dense2 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 1.0],
               [2.0, 2.0, 1.0],
               [1.0, 1.0, 2.0]]> : tensor<3x3xf32>
    %2 = dia.matmul ins(%dia2, %dense2: tensor<2x3xf32>, tensor<3x3xf32>)
                    outs(%r1 : tensor<?x3xf32>) -> tensor<?x3xf32>

    %dim2 = tensor.dim %2, %c0 : tensor<?x3xf32>
    %memref2 = memref.alloc(%dim2) : memref<?x3xf32>
    bufferization.materialize_in_destination %2 in writable %memref2
        : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast2 = memref.cast %memref2 : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast2) : (memref<*xf32>) -> ()
    memref.dealloc %memref2 : memref<?x3xf32>

    // =========================================================================
    // Variation 3: Upper Bandwidth Present (Lower=0, Upper=1)
    // =========================================================================
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK-NEXT: {{\[}}[3{{.*}}, 7{{.*}}, 7{{.*}}],
    // CHECK-NEXT:  [5{{.*}}, 7{{.*}}, 9{{.*}}],
    // CHECK-NEXT:  [3{{.*}}, 3{{.*}}, 3{{.*}}]]

    %r2 = tensor.empty(%c3) : tensor<?x3xf32>
    %dia3 = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 3.0],
               [2.0, 3.0, 0.0]]> : tensor<2x3xf32>
    %dense3 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 3.0, 1.0],
               [1.0, 2.0, 3.0],
               [1.0, 1.0, 1.0]]> : tensor<3x3xf32>

    %3 = dia.matmul ins(%dia3, %dense3: tensor<2x3xf32>, tensor<3x3xf32>)
                    outs(%r2 : tensor<?x3xf32>) -> tensor<?x3xf32>

    %dim3 = tensor.dim %3, %c0 : tensor<?x3xf32>
    %memref3 = memref.alloc(%dim3) : memref<?x3xf32>
    bufferization.materialize_in_destination %3 in writable %memref3
        : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast3 = memref.cast %memref3 : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast3) : (memref<*xf32>) -> ()
    memref.dealloc %memref3 : memref<?x3xf32>

    // =========================================================================
    // Variation 4: Dense B has no lower bandwidth (Lower=0, Upper=2)
    // =========================================================================
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK-NEXT: {{\[}}[2{{.*}}, 8{{.*}}, 3{{.*}}],
    // CHECK-NEXT:  [2{{.*}}, 12{{.*}}, 6{{.*}}],
    // CHECK-NEXT:  [0{{.*}}, 6{{.*}}, 7{{.*}}]]

    %r3 = tensor.empty(%c3) : tensor<?x3xf32>
    %dia4 = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 2.0, 3.0],
               [2.0, 3.0, 4.0],
               [1.0, 1.0, 0.0]]> : tensor<3x3xf32>
    %dense4 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 3.0, 1.0],
               [1.0, 2.0, 1.0],
               [1.0, 1.0, 1.0]]> : tensor<3x3xf32>

    %4 = dia.matmul ins(%dia4, %dense4: tensor<3x3xf32>, tensor<3x3xf32>)
                    outs(%r3 : tensor<?x3xf32>) -> tensor<?x3xf32>

    %dim4 = tensor.dim %4, %c0 : tensor<?x3xf32>
    %memref4 = memref.alloc(%dim4) : memref<?x3xf32>
    bufferization.materialize_in_destination %4 in writable %memref4
        : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast4 = memref.cast %memref4 : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast4) : (memref<*xf32>) -> ()
    memref.dealloc %memref4 : memref<?x3xf32>

    // =========================================================================
    // Variation 5: Dense B has no upper bandwidth (Lower=2, Upper=0)
    // =========================================================================
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK-NEXT: {{\[}}[3{{.*}}, 2{{.*}}, 0{{.*}}],
    // CHECK-NEXT:  [6{{.*}}, 7{{.*}}, 2{{.*}}],
    // CHECK-NEXT:  [7{{.*}}, 10{{.*}}, 8{{.*}}]]

    %r4 = tensor.empty(%c3) : tensor<?x3xf32>
    %dia5 = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 2.0, 3.0],
               [2.0, 3.0, 4.0],
               [1.0, 1.0, 0.0]]> : tensor<3x3xf32>
    %dense5 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 3.0, 1.0],
               [1.0, 2.0, 1.0],
               [1.0, 1.0, 2.0]]> : tensor<3x3xf32>

    %5 = dia.matmul ins(%dia5, %dense5: tensor<3x3xf32>, tensor<3x3xf32>)
                    outs(%r4 : tensor<?x3xf32>) -> tensor<?x3xf32>

    %memref5 = memref.alloc(%dim) : memref<?x3xf32>
    bufferization.materialize_in_destination %5 in writable %memref5
        : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast5 = memref.cast %memref5 : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast5) : (memref<*xf32>) -> ()
    memref.dealloc %memref5 : memref<?x3xf32>

    return
  }
}
