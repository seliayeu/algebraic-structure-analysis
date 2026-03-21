// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

// Chain matmul: I0(bw=2) * I1(bw=2) * I2(bw=2) * I3(bw=2)
// Result is DIA tensor 9x5 (lowerBw=4+4=8, upperBw=4+4=8, clamped to N-1=4)
// CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [9, 5]
// CHECK: {{\[}}[0{{.*}}, 0{{.*}}, 0{{.*}}, 0{{.*}}, 27{{.*}}],
// CHECK-NEXT: [0{{.*}}, 0{{.*}}, 0{{.*}}, 38{{.*}}, 38{{.*}}],
// CHECK-NEXT: [0{{.*}}, 0{{.*}}, 47{{.*}}, 52{{.*}}, 47{{.*}}],
// CHECK-NEXT: [0{{.*}}, 41{{.*}}, 63{{.*}}, 63{{.*}}, 41{{.*}}],
// CHECK-NEXT: [32{{.*}}, 54{{.*}}, 75{{.*}}, 54{{.*}}, 32{{.*}}],
// CHECK-NEXT: [41{{.*}}, 63{{.*}}, 63{{.*}}, 41{{.*}}, 0{{.*}}],
// CHECK-NEXT: [47{{.*}}, 52{{.*}}, 47{{.*}}, 0{{.*}}, 0{{.*}}],
// CHECK-NEXT: [38{{.*}}, 38{{.*}}, 0{{.*}}, 0{{.*}}, 0{{.*}}],
// CHECK-NEXT: [27{{.*}}, 0{{.*}}, 0{{.*}}, 0{{.*}}, 0{{.*}}]]
// CHECK: 4.700000e+01
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() -> f32 {
    %I0 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<5x5xf32>
    %I1 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<5x5xf32>
    %I2 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<5x5xf32>
    %I3 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<5x5xf32>

    %c0 = arith.constant 0 : index
    %e0 = tensor.empty(%c0) : tensor<?x5xf32>
    %e1 = tensor.empty(%c0) : tensor<?x5xf32>
    %e2 = tensor.empty(%c0) : tensor<?x5xf32>

    %R1 = dia.matmul ins(%I0, %I1 : tensor<5x5xf32>, tensor<5x5xf32>)
                     outs(%e0 : tensor<?x5xf32>) -> tensor<?x5xf32>
    %R2 = dia.matmul ins(%R1, %I2 : tensor<?x5xf32>, tensor<5x5xf32>)
                     outs(%e1 : tensor<?x5xf32>) -> tensor<?x5xf32>
    %R3 = dia.matmul ins(%R2, %I3 : tensor<?x5xf32>, tensor<5x5xf32>)
                     outs(%e2 : tensor<?x5xf32>) -> tensor<?x5xf32>

    %dim = tensor.dim %R3, %c0 : tensor<?x5xf32>
    %memref = memref.alloc(%dim) : memref<?x5xf32>
    bufferization.materialize_in_destination %R3 in writable %memref
        : (tensor<?x5xf32>, memref<?x5xf32>) -> ()
    %cast = memref.cast %memref : memref<?x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<?x5xf32>

    %index = arith.constant 5: index
    %result = tensor.extract %R3[%index, %index] : tensor<?x5xf32>
    return %result : f32
  }
}
