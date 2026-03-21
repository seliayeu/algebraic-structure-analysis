// // RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: arith.constant {metadata = {lowerBw = 2 : i64, propertyDims = [0, 1], upperBw = 2 : i64}} dense<1.000000e+00> : tensor<1024x1024xf32>
// CHECK: tensor.empty() {{.*}} : tensor<9x1024xf32>
// CHECK: tensor.empty() {{.*}} : tensor<13x1024xf32>
// CHECK: tensor.empty() {{.*}} : tensor<17x1024xf32>
// CHECK: dia.matmul {{.*}} -> tensor<9x1024xf32> {metadata = {dia = true, lowerBw = 4 : i64, propertyDims = [0, 1], upperBw = 4 : i64}}
// CHECK: dia.matmul {{.*}} -> tensor<13x1024xf32> {metadata = {dia = true, lowerBw = 6 : i64, propertyDims = [0, 1], upperBw = 6 : i64}}
// CHECK: dia.matmul {{.*}} -> tensor<17x1024xf32> {metadata = {dia = true, lowerBw = 8 : i64, propertyDims = [0, 1], upperBw = 8 : i64}}
// CHECK-NOT: tensor<?x1024xf32>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() -> f32 {
    %I0 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<1024x1024xf32>
    %I1 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<1024x1024xf32>
    %I2 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<1024x1024xf32>
    %I3 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<1024x1024xf32>

    %c0 = arith.constant 0 : index
    %e0 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e1 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e2 = tensor.empty(%c0) : tensor<?x1024xf32>

    %R1 = dia.matmul ins(%I0, %I1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%e0 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
    %R2 = dia.matmul ins(%R1, %I2 : tensor<?x1024xf32>, tensor<1024x1024xf32>)
                     outs(%e1 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
    %R3 = dia.matmul ins(%R2, %I3 : tensor<?x1024xf32>, tensor<1024x1024xf32>)
                     outs(%e2 : tensor<?x1024xf32>) -> tensor<?x1024xf32>

    %result = tensor.extract %R3[%c0, %c0] : tensor<?x1024xf32>
    return %result : f32
  }
}
