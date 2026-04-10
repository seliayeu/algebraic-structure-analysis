// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=false" --banded-rewrite \
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
    
    %denseA = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  1.0,  2.0,  3.0,  0.0],
               [ 0.0,  4.0,  1.0,  2.0,  3.0],
               [ 0.0,  0.0,  4.0,  1.0,  2.0],
               [ 0.0,  0.0,  0.0,  4.0,  1.0]]> : tensor<5x5xf32>

    %diaB = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0, 0.0, 6.0, 10.0, 14.0], // L2
               [ 0.0, 3.0,  7.0, 11.0, 15.0], // L1
               [ 1.0,  4.0,  8.0, 12.0, 16.0], // M0
               [ 2.0,  5.0,  9.0, 13.0,  0.0]]> : tensor<4x5xf32> // U1
               

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 0, 6, 10, 14],
    // CHECK-NEXT:  [0, 7, 11, 15, 19],
    // CHECK-NEXT:  [2, 5, 9, 13, 17],
    // CHECK-NEXT:  [4, 7, 11, 15, 0],
    // CHECK-NEXT:  [3, 3, 3, 0, 0]]
    %add_res = dia.elementwise kind = <add> ins(%denseA, %diaB : tensor<5x5xf32>, tensor<4x5xf32>) 
                                            outs(%0 : tensor<5x5xf32>) -> tensor<5x5xf32>
                                      
    %memref = memref.alloc() : memref<5x5xf32>
    bufferization.materialize_in_destination %add_res in %memref {writable} : (tensor<5x5xf32>, memref<5x5xf32>) -> ()
    %cast = memref.cast %memref : memref<5x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<5x5xf32>

    return
  }

  func.func @test_sub() {
    %0 = tensor.empty() : tensor<5x5xf32>
    
    %denseA = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  1.0,  2.0,  3.0,  0.0],
               [ 0.0,  4.0,  1.0,  2.0,  3.0],
               [ 0.0,  0.0,  4.0,  1.0,  2.0],
               [ 0.0,  0.0,  0.0,  4.0,  1.0]]> : tensor<5x5xf32>

    %diaB = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0, 0.0, 6.0, 10.0, 14.0], // L2
               [ 0.0, 3.0,  7.0, 11.0, 15.0], // L1
               [ 1.0,  4.0,  8.0, 12.0, 16.0], // M0
               [ 2.0,  5.0,  9.0, 13.0,  0.0]]> : tensor<4x5xf32> // U1

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 0, -6, -10, -14],
    // CHECK-NEXT:  [0, 1, -3, -7, -11],
    // CHECK-NEXT:  [0, -3, -7, -11, -15],
    // CHECK-NEXT:  [0, -3, -7, -11, 0],
    // CHECK-NEXT:  [3, 3, 3, 0, 0]]
    %sub_res = dia.elementwise kind = <sub> ins(%denseA, %diaB : tensor<5x5xf32>, tensor<4x5xf32>) 
                                            outs(%0 : tensor<5x5xf32>) -> tensor<5x5xf32>
                                      
    %memref = memref.alloc() : memref<5x5xf32>
    bufferization.materialize_in_destination %sub_res in %memref {writable} : (tensor<5x5xf32>, memref<5x5xf32>) -> ()
    %cast = memref.cast %memref : memref<5x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<5x5xf32>

    return
  }

  func.func @test_mul() {
    %0 = tensor.empty() : tensor<3x5xf32>
    
    %denseA = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  1.0,  2.0,  3.0,  0.0],
               [ 0.0,  4.0,  1.0,  2.0,  3.0],
               [ 0.0,  0.0,  4.0,  1.0,  2.0],
               [ 0.0,  0.0,  0.0,  4.0,  1.0]]> : tensor<5x5xf32>

    %diaB = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0, 0.0, 6.0, 10.0, 14.0], // L2
               [ 0.0, 3.0,  7.0, 11.0, 15.0], // L1
               [ 1.0,  4.0,  8.0, 12.0, 16.0], // M0
               [ 2.0,  5.0,  9.0, 13.0,  0.0]]> : tensor<4x5xf32> // U1

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [3, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 12, 28, 44, 60],
    // CHECK-NEXT:  [1, 4, 8, 12, 16],
    // CHECK-NEXT:  [4, 10, 18, 26, 0]]
    %mul_res = dia.elementwise kind = <mul> ins(%denseA, %diaB : tensor<5x5xf32>, tensor<4x5xf32>) 
                                            outs(%0 : tensor<3x5xf32>) -> tensor<3x5xf32>
                                      
    %memref = memref.alloc() : memref<3x5xf32>
    bufferization.materialize_in_destination %mul_res in %memref {writable} : (tensor<3x5xf32>, memref<3x5xf32>) -> ()
    %cast = memref.cast %memref : memref<3x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<3x5xf32>

    return
  }

  func.func @test_batch() {
    %0 = tensor.empty() : tensor<2x5x5xf32>
    
    // Batch 0: Original values. Batch 1: Original values * 2.
    // Note: propertyDims updated to [1, 2] to account for the leading batch dimension.
    %denseA = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} 
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

    %diaB = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
        dense<[[[ 0.0,  0.0,  6.0, 10.0, 14.0],
                [ 0.0,  3.0,  7.0, 11.0, 15.0],
                [ 1.0,  4.0,  8.0, 12.0, 16.0],
                [ 2.0,  5.0,  9.0, 13.0,  0.0]],
               [[ 0.0,  0.0, 12.0, 20.0, 28.0],
                [ 0.0,  6.0, 14.0, 22.0, 30.0],
                [ 2.0,  8.0, 16.0, 24.0, 32.0],
                [ 4.0, 10.0, 18.0, 26.0,  0.0]]]> : tensor<2x4x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [2, 5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[\[}}0, 0, 6, 10, 14],
    // CHECK-NEXT:  [0, 7, 11, 15, 19],
    // CHECK-NEXT:  [2, 5, 9, 13, 17],
    // CHECK-NEXT:  [4, 7, 11, 15, 0],
    // CHECK-NEXT:  [3, 3, 3, 0, 0]],
    // CHECK-NEXT:  {{\[\[}}0, 0, 12, 20, 28],
    // CHECK-NEXT:  [0, 14, 22, 30, 38],
    // CHECK-NEXT:  [4, 10, 18, 26, 34],
    // CHECK-NEXT:  [8, 14, 22, 30, 0],
    // CHECK-NEXT:  [6, 6, 6, 0, 0]]]
    %add_res = dia.elementwise kind = <add> ins(%denseA, %diaB : tensor<2x5x5xf32>, tensor<2x4x5xf32>) 
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
    call @test_mul() : () -> ()
    call @test_batch() : () -> ()
    return
  }
}
