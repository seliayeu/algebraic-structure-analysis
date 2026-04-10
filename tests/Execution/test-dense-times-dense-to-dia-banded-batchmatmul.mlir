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

  // ---------------------------------------------------------------------------
  // Combination 1: dense (L2, U0) * dense (L2, U0) = dia (batch x 4 x N)
  // (L=4 clamped to L=3 for 4x4 matrix)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}0,    0,    0,    2],
  // CHECK-NEXT:      [0,    0,    3,    3],
  // CHECK-NEXT:      [0,    2,    2,    2],
  // CHECK-NEXT:      [1,    1,    1,    1]],
  // CHECK-NEXT:     {{\[\[}}0,    0,    0,    2],
  // CHECK-NEXT:      [0,    0,    3,    3],
  // CHECK-NEXT:      [0,    2,    2,    2],
  // CHECK-NEXT:      [1,    1,    1,    1]]]
  func.func @test_c1_l2u0_l2u0() {
    %A = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>
    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 2: dense (L0, U2) * dense (L0, U2) = dia (batch x 4 x N)
  // (U=4 clamped to U=3 for 4x4 matrix)
  // ---------------------------------------------------------------------------
// CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}1,    1,    1,    1],
  // CHECK-NEXT:      [2,    2,    2,    0],
  // CHECK-NEXT:      [3,    3,    0,    0],
  // CHECK-NEXT:      [2,    0,    0,    0]],
  // CHECK-NEXT:     {{\[\[}}1,    1,    1,    1],
  // CHECK-NEXT:      [2,    2,    2,    0],
  // CHECK-NEXT:      [3,    3,    0,    0],
  // CHECK-NEXT:      [2,    0,    0,    0]]]
  func.func @test_c2_l0u2_l0u2() {
    %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>
    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 3: dense (L2, U0) * dense (L0, U2) = dia (batch x 5 x N)
  // (L=2, U=2 -> 5 diagonals, no clamping needed)
  // ---------------------------------------------------------------------------
// CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 5, 4]
  // CHECK-NEXT:    {{\[\[\[}}0,    0,    3,    3],
  // CHECK-NEXT:      [0,    2,    2,    2],
  // CHECK-NEXT:      [1,    1,    1,    1],
  // CHECK-NEXT:      [0,    0,    0,    0],
  // CHECK-NEXT:      [0,    0,    0,    2]],
  // CHECK-NEXT:     {{\[\[}}0,    0,    3,    3],
  // CHECK-NEXT:      [0,    2,    2,    2],
  // CHECK-NEXT:      [1,    1,    1,    1],
  // CHECK-NEXT:      [0,    0,    0,    0],
  // CHECK-NEXT:      [0,    0,    0,    0]]]
  func.func @test_c3_l2u0_l0u2() {
    %A = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x5x4xf32>
    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x5x4xf32>) -> tensor<2x5x4xf32>

    %memref = memref.alloc() : memref<2x5x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x5x4xf32>, memref<2x5x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x5x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x5x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 4: dense (L0, U1) * dense (L1, U0) = dia (batch x 3 x N)
  // (L=1, U=1 -> 3 diagonals, no clamping needed)
  // ---------------------------------------------------------------------------
// CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 3, 4]
  // CHECK-NEXT:    {{\[\[\[}}0,    0,    0,    0],
  // CHECK-NEXT:      [1,    1,    1,    1],
  // CHECK-NEXT:      [2,    2,    2,    0]],
  // CHECK-NEXT:     {{\[\[}}1,    1,    0,    0],
  // CHECK-NEXT:      [1,    1,    1,    1],
  // CHECK-NEXT:      [2,    2,    2,    0]]]
  func.func @test_c4_l0u1_l1u0() {
    %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x3x4xf32>
    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x3x4xf32>) -> tensor<2x3x4xf32>

    %memref = memref.alloc() : memref<2x3x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x3x4xf32>, memref<2x3x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x3x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x3x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 5: dense (L2, U2) * dense (L2, U2) = dia (batch x 7 x N)
  // (L=4 clamped to L=3, U=4 clamped to U=3 for 4x4 matrix)
  // ---------------------------------------------------------------------------
// CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 7, 4]
  // CHECK-NEXT:    {{\[\[\[}}0,    0,    0,    2],
  // CHECK-NEXT:      [0,    0,    3,    3],
  // CHECK-NEXT:      [0,    3,    4,    3],
  // CHECK-NEXT:      [3,    4,    4,    3],
  // CHECK-NEXT:      [3,    4,    3,    0],
  // CHECK-NEXT:      [3,    3,    0,    0],
  // CHECK-NEXT:      [2,    0,    0,    0]],
  // CHECK-NEXT:     {{\[\[}}0,    0,    0,    2],
  // CHECK-NEXT:      [0,    0,    3,    3],
  // CHECK-NEXT:      [0,    3,    4,    3],
  // CHECK-NEXT:      [3,    4,    4,    3],
  // CHECK-NEXT:      [3,    4,    3,    0],
  // CHECK-NEXT:      [3,    3,    0,    0],
  // CHECK-NEXT:      [2,    0,    0,    0]]]
  func.func @test_c5_l2u2_l2u2() {
    %A = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x7x4xf32>
    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x7x4xf32>) -> tensor<2x7x4xf32>

    %memref = memref.alloc() : memref<2x7x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x7x4xf32>, memref<2x7x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x7x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x7x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 6: dense (L0, U0) * dense (L0, U0) = dia (batch x 1 x N)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 1, 4]
  // CHECK-NEXT:    {{\[\[\[}}1,    1,    1,    1]],
  // CHECK-NEXT:     {{\[\[}}1,    1,    1,    1]]]
  func.func @test_c6_l0u0_l0u0() {
    %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x1x4xf32>
    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x1x4xf32>) -> tensor<2x1x4xf32>

    %memref = memref.alloc() : memref<2x1x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x1x4xf32>, memref<2x1x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x1x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x1x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Main execution sequence
  // ---------------------------------------------------------------------------
  func.func @main() {
    call @test_c1_l2u0_l2u0() : () -> ()
    call @test_c2_l0u2_l0u2() : () -> ()
    call @test_c3_l2u0_l0u2() : () -> ()
    call @test_c4_l0u1_l1u0() : () -> ()
    call @test_c5_l2u2_l2u2() : () -> ()
    call @test_c6_l0u0_l0u0() : () -> ()
    return
  }
}
