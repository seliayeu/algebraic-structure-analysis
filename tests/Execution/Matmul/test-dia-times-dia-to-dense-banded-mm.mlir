// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite \
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

  func.func @l1u0_l0u1_forced() {
    %0 = tensor.empty() : tensor<3x3xf32>
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 0.0, 8.0],
               [0.0, 4.0, 8.0],
               [1.0, 5.0, 9.0]]> : tensor<3x3xf32>
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 5.0, 9.0],
               [2.0, 6.0, 0.0],
               [1.0, 0.0, 0.0]]> : tensor<3x3xf32>

    // diaA * diaB = dense
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK: {{\[}}[1{{.*}}, 2{{.*}}, 0{{.*}}],
    // CHECK-NEXT: [4{{.*}}, 33{{.*}}, 34{{.*}}],
    // CHECK-NEXT: [0{{.*}}, 56{{.*}}, 137{{.*}}]]
    %1 = dia.matmul ins(%diaA, %diaB: tensor<3x3xf32>, tensor<3x3xf32>)
                    outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32> {metadata = {lowerBw = 1: i64, propertyDims = [0, 1], upperBw = 1: i64}}

    %memref = memref.alloc() : memref<3x3xf32>
    bufferization.materialize_in_destination %1 in %memref {writable} : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
    %cast = memref.cast %memref : memref<3x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<3x3xf32>
    return
  }

  func.func @l1u0_l0u1() {
    %0 = tensor.empty() : tensor<3x3xf32>
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 4.0, 8.0],
               [1.0, 5.0, 9.0]]> : tensor<2x3xf32>
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 5.0, 9.0],
               [2.0, 6.0, 0.0]]> : tensor<2x3xf32>

    // diaA * diaB = dense
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK: {{\[}}[1{{.*}}, 2{{.*}}, 0{{.*}}],
    // CHECK-NEXT: [4{{.*}}, 33{{.*}}, 30{{.*}}],
    // CHECK-NEXT: [0{{.*}}, 40{{.*}}, 129{{.*}}]]
    %1 = dia.matmul ins(%diaA, %diaB: tensor<2x3xf32>, tensor<2x3xf32>)
                    outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %memref = memref.alloc() : memref<3x3xf32>
    bufferization.materialize_in_destination %1 in %memref {writable} : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
    %cast = memref.cast %memref : memref<3x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<3x3xf32>
    return
  }

  func.func @l1u0_l1u0() {
    // diaC * diaC = dia
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 4]
    // CHECK: {{\[}}[0{{.*}}, 0{{.*}}, 4{{.*}}, 32{{.*}}],
    // CHECK-NEXT: [0{{.*}}, 6{{.*}}, 56{{.*}}, 152{{.*}}],
    // CHECK-NEXT: [1{{.*}}, 25{{.*}}, 81{{.*}}, 100{{.*}}]]
    %r1 = tensor.empty() : tensor<3x4xf32>
    %diaC = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 1.0, 4.0, 8.0],
               [1.0, 5.0, 9.0, 10.0]]> : tensor<2x4xf32>
    %2 = dia.matmul ins(%diaC, %diaC: tensor<2x4xf32>, tensor<2x4xf32>)
                    outs(%r1 : tensor<3x4xf32>) -> tensor<3x4xf32>
    %memref2 = memref.alloc() : memref<3x4xf32>
    bufferization.materialize_in_destination %2 in %memref2 {writable} : (tensor<3x4xf32>, memref<3x4xf32>) -> ()
    %cast2 = memref.cast %memref2 : memref<3x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast2) : (memref<*xf32>) -> ()
    memref.dealloc %memref2 : memref<3x4xf32>
    return
  }

  func.func @l1u0_l1u0_2() {
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 4.0, 8.0],
               [1.0, 5.0, 9.0]]> : tensor<2x3xf32>
    // diaA * diaA = dense
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK: {{\[}}[1{{.*}}, 0{{.*}}, 0{{.*}}],
    // CHECK-NEXT: [24{{.*}}, 25{{.*}}, 0{{.*}}],
    // CHECK-NEXT: [32{{.*}}, 112{{.*}}, 81{{.*}}]]
    %r2 = tensor.empty() : tensor<3x3xf32>
    %3 = dia.matmul ins(%diaA, %diaA: tensor<2x3xf32>, tensor<2x3xf32>)
                    outs(%r2 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %memref3 = memref.alloc() : memref<3x3xf32>
    bufferization.materialize_in_destination %3 in %memref3 {writable} : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
    %cast3 = memref.cast %memref3 : memref<3x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast3) : (memref<*xf32>) -> ()
    memref.dealloc %memref3 : memref<3x3xf32>
    return
  }

  func.func @l0u1_l0u1() {
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
      dense<[[1.0, 5.0, 9.0],
              [2.0, 6.0, 0.0]]> : tensor<2x3xf32>

    // diaB * diaB = dense
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3]
    // CHECK: {{\[}}[1{{.*}}, 12{{.*}}, 12{{.*}}],
    // CHECK-NEXT: [0{{.*}}, 25{{.*}}, 84{{.*}}],
    // CHECK-NEXT: [0{{.*}}, 0{{.*}}, 81{{.*}}]]
    %r3 = tensor.empty() : tensor<3x3xf32>
    %4 = dia.matmul ins(%diaB, %diaB: tensor<2x3xf32>, tensor<2x3xf32>)
                    outs(%r3 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %memref4 = memref.alloc() : memref<3x3xf32>
    bufferization.materialize_in_destination %4 in %memref4 {writable} : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
    %cast4 = memref.cast %memref4 : memref<3x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast4) : (memref<*xf32>) -> ()
    memref.dealloc %memref4 : memref<3x3xf32>
    return
  }

  func.func @main() {
    call @l1u0_l0u1_forced() : () -> ()
    call @l1u0_l0u1() : () -> ()
    call @l1u0_l1u0() : () -> ()
    call @l1u0_l1u0_2() : () -> ()
    call @l0u1_l0u1() : () -> ()
    return

  }
}
