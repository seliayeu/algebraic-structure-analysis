// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @forced() {
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [1, 3]
    // CHECK: {{\[}}[5{{.*}}, 4{{.*}}, 9{{.*}}]]
    %c0 = arith.constant 0 : index
    %dia = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[5.0, 2.0, 3.0]]> : tensor<1x3xf32>
    %dense = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 1.0],
               [1.0, 2.0, 2.0],
               [1.0, 1.0, 3.0]]> : tensor<3x3xf32>

    %c3 = arith.constant 3 : index

    %r0 = tensor.empty(%c3) : tensor<?x3xf32>
    %1 = dia.matmul ins(%dia, %dense: tensor<1x3xf32>, tensor<3x3xf32>)
                    outs(%r0 : tensor<?x3xf32>) -> tensor<?x3xf32> {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}

    %dim = tensor.dim %1, %c0 : tensor<?x3xf32>
    %memref = memref.alloc(%dim) : memref<?x3xf32>
    bufferization.materialize_in_destination %1 in writable %memref
        : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast = memref.cast %memref : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<?x3xf32>

    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [1, 3]
    // CHECK: {{\[}}[5{{.*}}, 4{{.*}}, 9{{.*}}]]
    %r1 = tensor.empty(%c3) : tensor<?x3xf32>
    %dense2 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 1.0],
               [1.0, 2.0, 1.0],
               [1.0, 2.0, 3.0]]> : tensor<3x3xf32>
    %2 = dia.matmul ins(%dia, %dense2: tensor<1x3xf32>, tensor<3x3xf32>)
                    outs(%r1 : tensor<?x3xf32>) -> tensor<?x3xf32> {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}

    %memref2 = memref.alloc(%dim) : memref<?x3xf32>
    bufferization.materialize_in_destination %2 in %memref2 {writable} : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast2 = memref.cast %memref2 : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast2) : (memref<*xf32>) -> ()
    memref.dealloc %memref2 : memref<?x3xf32>

    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [1, 3]
    // CHECK: {{\[}}[6{{.*}}, 8{{.*}}, 9{{.*}}]]
    %r2 = tensor.empty(%c3) : tensor<?x3xf32>
    %dia2 = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[5.0, 2.0, 3.0],
               [1.0, 2.0, 0.0]]> : tensor<2x3xf32>
    %3 = dia.matmul ins(%dia2, %dense2: tensor<2x3xf32>, tensor<3x3xf32>)
                    outs(%r2 : tensor<?x3xf32>) -> tensor<?x3xf32> {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}

    %memref3 = memref.alloc(%dim) : memref<?x3xf32>
    bufferization.materialize_in_destination %3 in %memref3 {writable} : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast3 = memref.cast %memref3 : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast3) : (memref<*xf32>) -> ()
    memref.dealloc %memref3 : memref<?x3xf32>
    return
  }

  func.func @regular() {
    // dia(lowerBw=0, upperBw=0) * dense(lowerBw=0, upperBw=1) = dia
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [2, 3]
    // CHECK: {{\[}}[5{{.*}}, 4{{.*}}, 9{{.*}}],
    // CHECK-NEXT: [5{{.*}}, 4{{.*}}, 0{{.*}}]]
    %c0 = arith.constant 0 : index
    %dia = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[5.0, 2.0, 3.0]]> : tensor<1x3xf32>
    %dense = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 1.0],
               [1.0, 2.0, 2.0],
               [1.0, 1.0, 3.0]]> : tensor<3x3xf32>

    %c3 = arith.constant 3 : index

    %r0 = tensor.empty(%c3) : tensor<?x3xf32>
    %1 = dia.matmul ins(%dia, %dense: tensor<1x3xf32>, tensor<3x3xf32>)
                    outs(%r0 : tensor<?x3xf32>) -> tensor<?x3xf32> 

    %dim = tensor.dim %1, %c0 : tensor<?x3xf32>
    %memref = memref.alloc(%dim) : memref<?x3xf32>
    bufferization.materialize_in_destination %1 in writable %memref
        : (tensor<?x3xf32>, memref<?x3xf32>) -> ()
    %cast = memref.cast %memref : memref<?x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<?x3xf32>

    // dia(lowerBw=0, upperBw=0) * dense(lowerBw=1, upperBw=0) = dia
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [2, 3]
    // CHECK: {{\[}}[0{{.*}}, 2{{.*}}, 6{{.*}}],
    // CHECK-NEXT: [5{{.*}}, 4{{.*}}, 9{{.*}}]]
    %r1 = tensor.empty() : tensor<2x3xf32>
    %dense2 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 1.0],
               [1.0, 2.0, 1.0],
               [1.0, 2.0, 3.0]]> : tensor<3x3xf32>
    %2 = dia.matmul ins(%dia, %dense2: tensor<1x3xf32>, tensor<3x3xf32>)
                    outs(%r1 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %memref2 = memref.alloc() : memref<2x3xf32>
    bufferization.materialize_in_destination %2 in %memref2 {writable} : (tensor<2x3xf32>, memref<2x3xf32>) -> ()
    %cast2 = memref.cast %memref2 : memref<2x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast2) : (memref<*xf32>) -> ()
    memref.dealloc %memref2 : memref<2x3xf32>

    // dia(lowerBw=0, upperBw=1) * dense(lowerBw=1, upperBw=0) = dia
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK: {{\[}}[0{{.*}}, 2{{.*}}, 6{{.*}}],
    // CHECK-NEXT: [6{{.*}}, 8{{.*}}, 9{{.*}}],
    // CHECK-NEXT: [2{{.*}}, 6{{.*}}, 0{{.*}}]]
    %r2 = tensor.empty() : tensor<3x3xf32>
    %dia2 = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[5.0, 2.0, 3.0],
               [1.0, 2.0, 0.0]]> : tensor<2x3xf32>
    %3 = dia.matmul ins(%dia2, %dense2: tensor<2x3xf32>, tensor<3x3xf32>)
                    outs(%r2 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %memref3 = memref.alloc() : memref<3x3xf32>
    bufferization.materialize_in_destination %3 in %memref3 {writable} : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
    %cast3 = memref.cast %memref3 : memref<3x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast3) : (memref<*xf32>) -> ()
    memref.dealloc %memref3 : memref<3x3xf32>
    return
  }

  func.func @main() {

    call @forced() : () -> ()
    call @regular() : () -> ()

    return
  }
}
