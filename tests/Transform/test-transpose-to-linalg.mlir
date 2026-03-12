// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite | FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @banded_transpose
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map1]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: linalg.yield

// CHECK-LABEL: func.func @diag_transpose
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.generic
// CHECK-NOT: iterator_types = ["parallel", "parallel"]
// CHECK-NOT: linalg.yield

module {
  func.func @banded_transpose() -> tensor<3x3xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,1.0,0.0],[0.0,4.0,2.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %1 = linalg.transpose ins(%A : tensor<3x3xf32>)
                          outs(%0 : tensor<3x3xf32>)
                          permutation = [1, 0]
    return %1 : tensor<3x3xf32>
  }

  func.func @diag_transpose() -> tensor<3x3xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %1 = linalg.transpose ins(%A : tensor<3x3xf32>)
                          outs(%0 : tensor<3x3xf32>)
                          permutation = [1, 0]
    return %1 : tensor<3x3xf32>
  }
}
