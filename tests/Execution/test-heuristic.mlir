// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-analysis="heuristic=true" --canonicalize --banded-rewrite \
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

  // CHECK:      Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [0,   3,  12,  27]
  // CHECK-NEXT: [1,   8,  21,   4]
  // CHECK-NEXT: [4,  15,  32,   0]
  func.func @insert_from_dense_lhs() {
    %dense_lhs = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 0.0, 0.0],
               [3.0, 4.0, 5.0, 0.0],
               [0.0, 6.0, 7.0, 8.0],
               [0.0, 0.0, 9.0, 1.0]]> : tensor<4x4xf32>
    %dia_rhs = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
    %out = tensor.empty() : tensor<3x4xf32>
    %result = dia.matmul ins(%dense_lhs, %dia_rhs : tensor<4x4xf32>, tensor<1x4xf32>)
                         outs(%out : tensor<3x4xf32>) -> tensor<3x4xf32>
    %alloc = memref.alloc() : memref<3x4xf32>
    bufferization.materialize_in_destination %result in writable %alloc
        : (tensor<3x4xf32>, memref<3x4xf32>) -> ()
    %cast = memref.cast %alloc : memref<3x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<3x4xf32>
    return
  }

  // CHECK:      Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [5, 4] strides = [4, 1]
  // CHECK-NEXT: [0,   0,  18,  54]
  // CHECK-NEXT: [0,  15,  66,  72]
  // CHECK-NEXT: [7,  52, 151,  73]
  // CHECK-NEXT: [10, 55,  64,   0]
  // CHECK-NEXT: [10, 40,   0,   0]
  func.func @insert_from_dense_both() {
    %dense_lhs = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 0.0, 0.0],
               [3.0, 4.0, 5.0, 0.0],
               [0.0, 6.0, 7.0, 8.0],
               [0.0, 0.0, 9.0, 1.0]]> : tensor<4x4xf32>
    %dense_rhs = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 0.0, 0.0],
               [3.0, 4.0, 5.0, 0.0],
               [0.0, 6.0, 7.0, 8.0],
               [0.0, 0.0, 9.0, 1.0]]> : tensor<4x4xf32>
    %out = tensor.empty() : tensor<5x4xf32>
    %result = dia.matmul ins(%dense_lhs, %dense_rhs : tensor<4x4xf32>, tensor<4x4xf32>)
                         outs(%out : tensor<5x4xf32>) -> tensor<5x4xf32>
    %alloc = memref.alloc() : memref<5x4xf32>
    bufferization.materialize_in_destination %result in writable %alloc
        : (tensor<5x4xf32>, memref<5x4xf32>) -> ()
    %cast = memref.cast %alloc : memref<5x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<5x4xf32>
    return
  }

  // CHECK:      Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [0,   5,  14,  24]
  // CHECK-NEXT: [1,   8,  21,   0]
  // CHECK-NEXT: [6,  18,   0,   0]
  func.func @no_from_dense_when_already_dia() {
    %dia_lhs = arith.constant {metadata = {dia = true, upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 5.0, 7.0, 8.0],
               [1.0, 4.0, 7.0, 0.0],
               [3.0, 6.0, 0.0, 0.0]]> : tensor<3x4xf32>
    %dia_rhs = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
    %out = tensor.empty() : tensor<3x4xf32>
    %result = dia.matmul ins(%dia_lhs, %dia_rhs : tensor<3x4xf32>, tensor<1x4xf32>)
                         outs(%out : tensor<3x4xf32>) -> tensor<3x4xf32>
    %alloc = memref.alloc() : memref<3x4xf32>
    bufferization.materialize_in_destination %result in writable %alloc
        : (tensor<3x4xf32>, memref<3x4xf32>) -> ()
    %cast = memref.cast %alloc : memref<3x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<3x4xf32>
    return
  }

  func.func @main() {
    call @insert_from_dense_lhs() : () -> ()
    call @insert_from_dense_both() : () -> ()
    call @no_from_dense_when_already_dia() : () -> ()
    return
  }
}
