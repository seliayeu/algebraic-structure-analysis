// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-NOT: linalg.matmul

// CHECK: scf.for
// CHECK-SAME: iter_args

// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: tensor.extract

// CHECK: scf.for
// CHECK-SAME: iter_args

// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor.insert
// CHECK: scf.yield

// CHECK: scf.yield

// CHECK: scf.yield

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %0 = tensor.empty() : tensor<4x4xf32>
    %A = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
            dense<[[1.0, 4.0, 0.0, 0.0], [3.0, 4.0, 1.0, 0.0],[0.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 3.0]]> : tensor<4x4xf32>
    %3 = linalg.matmul ins(%A, %A : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %memref = memref.alloc() : memref<4x4xf32>
    bufferization.materialize_in_destination %3 in %memref {writable}
        : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
    %cast = memref.cast %memref : memref<4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<4x4xf32>
    return
  }
}
