// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite

// CHECK-LABEL: func.func @main
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK: scf.for %[[DC:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:   arith.subi %[[DC]], %{{.*}}
// CHECK:   scf.for %[[ROW:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK:     arith.index_cast %[[ROW]] : index to i64
// CHECK:     arith.cmpi sge, %{{.*}}, %[[C0]]
// CHECK:     arith.cmpi slt, %{{.*}}, %[[C3]]
// CHECK:     arith.andi
// CHECK:     scf.if %{{.*}}
// CHECK:       arith.maxsi
// CHECK:       arith.minsi
// CHECK:       scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %[[C1]]
// CHECK:         arith.addi %{{.*}}, %[[C0_I64]]
// CHECK:         tensor.extract %{{.*}}[%{{.*}}, %[[ROW]]]
// CHECK:         tensor.extract %{{.*}}[%[[K]], %{{.*}}]
// CHECK:         tensor.extract %{{.*}}[%[[DC]], %[[ROW]]]
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:         tensor.insert %{{.*}} into %{{.*}}[%[[DC]], %[[ROW]]]
// CHECK-NOT: dia.matmul
module {
  func.func @main() -> tensor<2x3xf32> {
    %r0 = tensor.empty() : tensor<2x3xf32>
    %dia = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[5.0, 2.0, 3.0]]> : tensor<1x3xf32>
    %dense = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 0.0],
               [0.0, 2.0, 2.0],
               [0.0, 0.0, 3.0]]> : tensor<3x3xf32>
    %1 = dia.matmul ins(%dia, %dense: tensor<1x3xf32>, tensor<3x3xf32>)
                    outs(%r0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
