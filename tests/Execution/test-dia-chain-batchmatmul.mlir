// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

// CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
// CHECK:      {{\[\[\[}}0{{.*}}, 0{{.*}}, 1{{.*}}, 1{{.*}}],
// CHECK-NEXT:   [0{{.*}}, 3{{.*}}, 3{{.*}}, 3{{.*}}],
// CHECK-NEXT:   [2{{.*}}, 3{{.*}}, 3{{.*}}, 2{{.*}}],
// CHECK-NEXT:   [1{{.*}}, 1{{.*}}, 1{{.*}}, 0{{.*}}]],
// CHECK-NEXT:  {{\[\[}}0{{.*}}, 0{{.*}}, 4{{.*}}, 4{{.*}}],
// CHECK-NEXT:   [0{{.*}}, 12{{.*}}, 12{{.*}}, 12{{.*}}],
// CHECK-NEXT:   [8{{.*}}, 12{{.*}}, 12{{.*}}, 8{{.*}}],
// CHECK-NEXT:   [4{{.*}}, 4{{.*}}, 4{{.*}}, 0{{.*}}]]]

module {
  func.func private @printMemrefF32(memref<*xf32>)
  
  func.func @main() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} 
        dense<[[[0.0, 1.0, 1.0, 1.0],  // B0, L1 
               [1.0, 1.0, 1.0, 1.0]],  // B0, M0
              [[0.0, 2.0, 2.0, 2.0],   // B1, L1 
               [2.0, 2.0, 2.0, 2.0]]]> : tensor<2x2x4xf32> 
               
    %B = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
        dense<[[[1.0, 1.0, 1.0, 1.0],  // B0, M0
               [1.0, 1.0, 1.0, 0.0]],  // B0, U1 
              [[2.0, 2.0, 2.0, 2.0],   // B1, M0
               [2.0, 2.0, 2.0, 0.0]]]> : tensor<2x2x4xf32> 

    %C = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} 
        dense<[[[0.0, 1.0, 1.0, 1.0],  // B0, L1 
               [1.0, 1.0, 1.0, 1.0]],  // B0, M0
              [[0.0, 1.0, 1.0, 1.0],   // B1, L1 
               [1.0, 1.0, 1.0, 1.0]]]> : tensor<2x2x4xf32> 

    %c0_empty = arith.constant 0 : index
    %e0 = tensor.empty(%c0_empty) : tensor<2x?x4xf32>
    %e1 = tensor.empty(%c0_empty) : tensor<2x?x4xf32>

    %R1 = dia.batch_matmul ins(%A, %B : tensor<2x2x4xf32>, tensor<2x2x4xf32>) 
                           outs(%e0 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
                           
    %R2 = dia.batch_matmul ins(%R1, %C : tensor<2x?x4xf32>, tensor<2x2x4xf32>) 
                           outs(%e1 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>

    %c1 = arith.constant 1 : index
    %dim = tensor.dim %R2, %c1 : tensor<2x?x4xf32> 
    %memref = memref.alloc(%dim) : memref<2x?x4xf32>
    bufferization.materialize_in_destination %R2 in writable %memref 
        : (tensor<2x?x4xf32>, memref<2x?x4xf32>) -> ()
    
    %cast = memref.cast %memref : memref<2x?x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x?x4xf32>
    
    return
  }
}
