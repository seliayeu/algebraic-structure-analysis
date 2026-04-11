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

  func.func @test_add() {
    %0 = tensor.empty() : tensor<5x5xf32>
    
    %denseB = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  1.0,  2.0,  3.0,  0.0],
               [ 0.0,  4.0,  1.0,  2.0,  3.0],
               [ 0.0,  0.0,  4.0,  1.0,  2.0],
               [ 0.0,  0.0,  0.0,  4.0,  1.0]]> : tensor<5x5xf32>

    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0, 0.0, 6.0, 10.0, 14.0], // L2: 3 elements
               [ 0.0, 3.0,  7.0, 11.0, 15.0], // L1: 4 elements
               [ 1.0,  4.0,  8.0, 12.0, 16.0], // M0: 5 elements
               [ 2.0,  5.0,  9.0, 13.0,  0.0]]> : tensor<4x5xf32> // U1: 4 elements

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}2, 4, 3, 0, 0],
    // CHECK-NEXT:  [7, 5, 7, 3, 0],
    // CHECK-NEXT:  [6, 11, 9, 11, 3],
    // CHECK-NEXT:  [0, 10, 15, 13, 15],
    // CHECK-NEXT:  [0, 0, 14, 19, 17]]
    %add_res = dia.elementwise kind = <add> ins(%denseB, %diaA : tensor<5x5xf32>, tensor<4x5xf32>) 
                                            outs(%0 : tensor<5x5xf32>) -> tensor<5x5xf32>
                                      
    %memref_add = memref.alloc() : memref<5x5xf32>
    bufferization.materialize_in_destination %add_res in %memref_add {writable} : (tensor<5x5xf32>, memref<5x5xf32>) -> ()
    %cast_add = memref.cast %memref_add : memref<5x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast_add) : (memref<*xf32>) -> ()
    memref.dealloc %memref_add : memref<5x5xf32>

    return
  }

  func.func @test_sub() {
    %0 = tensor.empty() : tensor<5x5xf32>
    
    %denseB = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  1.0,  2.0,  3.0,  0.0],
               [ 0.0,  4.0,  1.0,  2.0,  3.0],
               [ 0.0,  0.0,  4.0,  1.0,  2.0],
               [ 0.0,  0.0,  0.0,  4.0,  1.0]]> : tensor<5x5xf32>

    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0, 0.0, 6.0, 10.0, 14.0], // L2: 3 elements
               [ 0.0, 3.0,  7.0, 11.0, 15.0], // L1: 4 elements
               [ 1.0,  4.0,  8.0, 12.0, 16.0], // M0: 5 elements
               [ 2.0,  5.0,  9.0, 13.0,  0.0]]> : tensor<4x5xf32> // U1: 4 elements

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 0, 3, 0, 0],
    // CHECK-NEXT:  [1, -3, -3, 3, 0],
    // CHECK-NEXT:  [-6, -3, -7, -7, 3],
    // CHECK-NEXT:  [0, -10, -7, -11, -11],
    // CHECK-NEXT:  [0, 0, -14, -11, -15]]
    %sub_res = dia.elementwise kind = <sub> ins(%denseB, %diaA : tensor<5x5xf32>, tensor<4x5xf32>) 
                                            outs(%0 : tensor<5x5xf32>) -> tensor<5x5xf32>
                                      
    %memref_sub = memref.alloc() : memref<5x5xf32>
    bufferization.materialize_in_destination %sub_res in %memref_sub {writable} : (tensor<5x5xf32>, memref<5x5xf32>) -> ()
    %cast_sub = memref.cast %memref_sub : memref<5x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast_sub) : (memref<*xf32>) -> ()
    memref.dealloc %memref_sub : memref<5x5xf32>

    return
  }

  func.func @test_batch() {
    %0 = tensor.empty() : tensor<2x5x5xf32>
    
    // Batch 0: Original values. Batch 1: Original values * 2.
    // Note: propertyDims updated to [1, 2] to account for the leading batch dimension.
    %denseB = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} 
        dense<[[[ 1.0,  2.0,  3.0,  0.0,  0.0],
                [ 4.0,  1.0,  2.0,  3.0,  0.0],
                [ 0.0,  4.0,  1.0,  2.0,  3.0],
                [ 0.0,  0.0,  4.0,  1.0,  2.0],
                [ 0.0,  0.0,  0.0,  4.0,  1.0]],
               [[ 2.0,  4.0,  6.0,  0.0,  0.0],
                [ 8.0,  2.0,  4.0,  6.0,  0.0],
                [ 0.0,  8.0,  2.0,  4.0,  6.0],
                [ 0.0,  0.0,  8.0,  2.0,  4.0],
                [ 0.0,  0.0,  0.0,  8.0,  2.0]]]> : tensor<2x5x5xf32>

    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
        dense<[[[ 0.0, 0.0, 6.0, 10.0, 14.0],
                [ 0.0, 3.0,  7.0, 11.0, 15.0],
                [ 1.0,  4.0,  8.0, 12.0, 16.0],
                [ 2.0,  5.0,  9.0, 13.0,  0.0]],
               [[ 0.0, 0.0, 12.0, 20.0, 28.0],
                [ 0.0, 6.0, 14.0, 22.0, 30.0],
                [ 2.0,  8.0, 16.0, 24.0, 32.0],
                [ 4.0, 10.0, 18.0, 26.0,  0.0]]]> : tensor<2x4x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [2, 5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[\[}}2, 4, 3, 0, 0],
    // CHECK-NEXT:  [7, 5, 7, 3, 0],
    // CHECK-NEXT:  [6, 11, 9, 11, 3],
    // CHECK-NEXT:  [0, 10, 15, 13, 15],
    // CHECK-NEXT:  [0, 0, 14, 19, 17]],
    // CHECK-NEXT:  {{\[\[}}4, 8, 6, 0, 0],
    // CHECK-NEXT:  [14, 10, 14, 6, 0],
    // CHECK-NEXT:  [12, 22, 18, 22, 6],
    // CHECK-NEXT:  [0, 20, 30, 26, 30],
    // CHECK-NEXT:  [0, 0, 28, 38, 34]]]
    %add_res = dia.elementwise kind = <add> ins(%denseB, %diaA : tensor<2x5x5xf32>, tensor<2x4x5xf32>) 
                                            outs(%0 : tensor<2x5x5xf32>) -> tensor<2x5x5xf32>
                                      
    %memref_add = memref.alloc() : memref<2x5x5xf32>
    bufferization.materialize_in_destination %add_res in %memref_add {writable} : (tensor<2x5x5xf32>, memref<2x5x5xf32>) -> ()
    %cast_add = memref.cast %memref_add : memref<2x5x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast_add) : (memref<*xf32>) -> ()
    memref.dealloc %memref_add : memref<2x5x5xf32>

    return
  }

  func.func @main() {
    call @test_add() : () -> ()
    call @test_sub() : () -> ()
    call @test_batch() : () -> ()
    return
  }
}
