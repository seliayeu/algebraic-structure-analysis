// RUN: %build/tools/alg-opt %s --banded-analysis --banded-lowering | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0, d0)>
// CHECK-LABEL: func.func @main
// CHECK-NOT: linalg.matmul
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map, #map]
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: ^bb0
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: linalg.yield

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %B = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[9.0,0.0,0.0],[0.0,9.0,0.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>
    %3 = linalg.matmul ins(%A, %B : tensor<3x3xf32>, tensor<3x3xf32>)
                       outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %memref = memref.alloc() : memref<3x3xf32>
    bufferization.materialize_in_destination %3 in %memref {writable}
        : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
    %cast = memref.cast %memref : memref<3x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<3x3xf32>
    return
  }
}
