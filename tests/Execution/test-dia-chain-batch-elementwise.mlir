// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

// CHECK:      Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 3, 4]
// CHECK-NEXT: {{\[\[\[}}1, 1, 1, 0],
// CHECK-NEXT:  [10, 10, 10, 10],
// CHECK-NEXT:  [0, 10, 10, 10]],
// CHECK-NEXT:  {{\[\[}}10, 10, 10, 0],
// CHECK-NEXT:  [-8, 32, 32, 32],
// CHECK-NEXT:  [0, -10, -10, -10]]]

module {
  func.func private @printMemrefF32(memref<*xf32>)
  
  func.func @main() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} 
        dense<[[[1.0, 1.0, 0.0, 0.0],  // L2
                [2.0, 2.0, 2.0, 0.0],  // L1
                [3.0, 3.0, 3.0, 3.0]], // M0
               [[2.0, 2.0, 0.0, 0.0], 
                [4.0, 4.0, 4.0, 0.0], 
                [6.0, 6.0, 6.0, 6.0]]]> : tensor<2x3x4xf32>
               
    %B = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} 
        dense<[[[4.0, 4.0, 4.0, 4.0],  // M0
                [0.0, 5.0, 5.0, 5.0],  // U1
                [0.0, 0.0, 6.0, 6.0]], // U2
               [[8.0, 8.0, 8.0, 8.0], 
                [0.0, 10.0, 10.0, 10.0], 
                [0.0, 0.0, 12.0, 12.0]]]> : tensor<2x3x4xf32>

    %C = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
        dense<[[[1.0, 1.0, 1.0, 0.0],  // L1
                [2.0, 2.0, 2.0, 2.0],  // M0
                [0.0, 3.0, 3.0, 3.0]], // U1
               [[2.0, 2.0, 2.0, 0.0], 
                [4.0, 4.0, 4.0, 4.0], 
                [0.0, 6.0, 6.0, 6.0]]]> : tensor<2x3x4xf32>

    %D = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
        dense<[[[1.0, 1.0, 1.0, 0.0],  // L1
                [4.0, 4.0, 4.0, 4.0],  // M0
                [0.0, 5.0, 5.0, 5.0]], // U1
               [[2.0, 2.0, 2.0, 0.0], 
                [8.0, 8.0, 8.0, 8.0], 
                [0.0, 10.0, 10.0, 10.0]]]> : tensor<2x3x4xf32>

    %c0 = arith.constant 0 : index
    
    %e0 = tensor.empty(%c0) : tensor<2x?x4xf32>
    %e1 = tensor.empty(%c0) : tensor<2x?x4xf32>
    %e2 = tensor.empty(%c0) : tensor<2x?x4xf32>

    %R1 = dia.elementwise kind = <add> ins(%A, %B : tensor<2x3x4xf32>, tensor<2x3x4xf32>) 
                                       outs(%e0 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
                                       
    %R2 = dia.elementwise kind = <mul> ins(%R1, %C : tensor<2x?x4xf32>, tensor<2x3x4xf32>) 
                                       outs(%e1 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>

    %R3 = dia.elementwise kind = <sub> ins(%R2, %D : tensor<2x?x4xf32>, tensor<2x3x4xf32>) 
                                       outs(%e2 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
                                       
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %R3, %c1 : tensor<2x?x4xf32>
    %memref = memref.alloc(%dim) : memref<2x?x4xf32>
    bufferization.materialize_in_destination %R3 in writable %memref 
        : (tensor<2x?x4xf32>, memref<2x?x4xf32>) -> ()
    
    %cast = memref.cast %memref : memref<2x?x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x?x4xf32>
    
    return
  }
}
