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

    %denseB = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 5.0,  6.0,  0.0,  0.0,  0.0],
               [ 7.0,  5.0,  6.0,  0.0,  0.0],
               [ 8.0,  7.0,  5.0,  6.0,  0.0],
               [ 0.0,  8.0,  7.0,  5.0,  6.0],
               [ 0.0,  0.0,  8.0,  7.0,  5.0]]> : tensor<5x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 0, 8, 8, 8],
    // CHECK-NEXT:  [0, 11, 11, 11, 11],
    // CHECK-NEXT:  [6, 6, 6, 6, 6],
    // CHECK-NEXT:  [8, 8, 8, 8, 0],
    // CHECK-NEXT:  [3, 3, 3, 0, 0]]
    %add_res = dia.elementwise kind = <add> ins(%denseA, %denseB : tensor<5x5xf32>, tensor<5x5xf32>) 
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

    %denseB = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 5.0,  6.0,  0.0,  0.0,  0.0],
               [ 7.0,  5.0,  6.0,  0.0,  0.0],
               [ 8.0,  7.0,  5.0,  6.0,  0.0],
               [ 0.0,  8.0,  7.0,  5.0,  6.0],
               [ 0.0,  0.0,  8.0,  7.0,  5.0]]> : tensor<5x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 0, -8, -8, -8],
    // CHECK-NEXT:  [0, -3, -3, -3, -3],
    // CHECK-NEXT:  [-4, -4, -4, -4, -4],
    // CHECK-NEXT:  [-4, -4, -4, -4, 0],
    // CHECK-NEXT:  [3, 3, 3, 0, 0]]
    %sub_res = dia.elementwise kind = <sub> ins(%denseA, %denseB : tensor<5x5xf32>, tensor<5x5xf32>) 
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

    %denseB = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 5.0,  6.0,  0.0,  0.0,  0.0],
               [ 7.0,  5.0,  6.0,  0.0,  0.0],
               [ 8.0,  7.0,  5.0,  6.0,  0.0],
               [ 0.0,  8.0,  7.0,  5.0,  6.0],
               [ 0.0,  0.0,  8.0,  7.0,  5.0]]> : tensor<5x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [3, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 28, 28, 28, 28],
    // CHECK-NEXT:  [5, 5, 5, 5, 5],
    // CHECK-NEXT:  [12, 12, 12, 12, 0]]
    %mul_res = dia.elementwise kind = <mul> ins(%denseA, %denseB : tensor<5x5xf32>, tensor<5x5xf32>) 
                               outs(%0 : tensor<3x5xf32>) -> tensor<3x5xf32>
                                      
    %memref = memref.alloc() : memref<3x5xf32>
    bufferization.materialize_in_destination %mul_res in %memref {writable} : (tensor<3x5xf32>, memref<3x5xf32>) -> ()
    %cast = memref.cast %memref : memref<3x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<3x5xf32>

    return
  }

  func.func @main() {
    call @test_add() : () -> ()
    call @test_sub() : () -> ()
    call @test_mul() : () -> ()
    return
  }
}
