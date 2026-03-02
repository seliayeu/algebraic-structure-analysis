// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @mul
// CHECK-NOT: linalg.elementwise_kind
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: arith.mulf
// CHECK: tensor.insert
// CHECK: scf.yield
// CHECK: scf.yield

// CHECK-LABEL: func.func @add
// CHECK-NOT: linalg.elementwise_kind
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: arith.addf
// CHECK: tensor.insert
// CHECK: scf.yield
// CHECK: scf.yield

// CHECK-LABEL: func.func @sub
// CHECK-NOT: linalg.elementwise_kind
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: arith.subf
// CHECK: tensor.insert
// CHECK: scf.yield
// CHECK: scf.yield

// CHECK-LABEL: func.func @diag_mul
// CHECK-NOT: linalg.elementwise_kind
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map, #map]
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: ^bb0
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: linalg.yield

// CHECK-LABEL: func.func @diag_add
// CHECK-NOT: linalg.elementwise_kind
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map, #map]
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: ^bb0
// CHECK-NEXT: arith.addf
// CHECK-NEXT: linalg.yield

// CHECK-LABEL: func.func @diag_sub
// CHECK-NOT: linalg.elementwise_kind
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map, #map]
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: ^bb0
// CHECK-NEXT: arith.subf
// CHECK-NEXT: linalg.yield





module {
  func.func @mul() -> tensor<3x3xf32> {

    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,1.0,0.0],[0.0,4.0,2.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>

    %B = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[9.0,3.0,0.0],[0.0,9.0,3.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>

    %3 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
      ins(%A, %B: tensor<3x3xf32>, tensor<3x3xf32>)
      outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>

    return %3: tensor<3x3xf32>
  }

  func.func @add() -> tensor<3x3xf32> {

    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,1.0,0.0],[0.0,4.0,2.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>

    %B = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[9.0,3.0,0.0],[0.0,9.0,3.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>

    %3 = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%A, %B: tensor<3x3xf32>, tensor<3x3xf32>)
      outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>

    return %3: tensor<3x3xf32>
  }

    func.func @sub() -> tensor<3x3xf32> {

    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,1.0,0.0],[0.0,4.0,2.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>

    %B = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[9.0,3.0,0.0],[0.0,9.0,3.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>

    %3 = linalg.elementwise kind=#linalg.elementwise_kind<sub>
      ins(%A, %B: tensor<3x3xf32>, tensor<3x3xf32>)
      outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>

    return %3: tensor<3x3xf32>
  }

  func.func @diag_mul() -> tensor<3x3xf32> {

    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,1.0,0.0],[0.0,4.0,2.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>

    %B = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[9.0,3.0,0.0],[0.0,9.0,3.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>

    %3 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
      ins(%A, %B: tensor<3x3xf32>, tensor<3x3xf32>)
      outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>

    return %3: tensor<3x3xf32>
  }

    func.func @diag_add() -> tensor<3x3xf32> {

    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,1.0,0.0],[0.0,4.0,2.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>

    %B = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[9.0,3.0,0.0],[0.0,9.0,3.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>

    %3 = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%A, %B: tensor<3x3xf32>, tensor<3x3xf32>)
      outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>

    return %3: tensor<3x3xf32>
  }

  func.func @diag_sub() -> tensor<3x3xf32> {

    %0 = tensor.empty() : tensor<3x3xf32>
    %A = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[2.0,1.0,0.0],[0.0,4.0,2.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>

    %B = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[9.0,3.0,0.0],[0.0,9.0,3.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>

    %3 = linalg.elementwise kind=#linalg.elementwise_kind<sub>
      ins(%A, %B: tensor<3x3xf32>, tensor<3x3xf32>)
      outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>

    return %3: tensor<3x3xf32>
  }



}
