// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

// CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [2, 5]
// CHECK: {{\[\[}}2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 0{{.*}}],
// CHECK-NEXT:  [20{{.*}}, 20{{.*}}, 20{{.*}}, 20{{.*}}, 20{{.*}}]]

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}} 
        dense<[[1.0, 1.0, 1.0, 1.0, 0.0],  // L1
               [2.0, 2.0, 2.0, 2.0, 2.0]]> : tensor<2x5xf32> // M0
               
    %B = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[4.0, 4.0, 4.0, 4.0, 4.0],  // M0
               [0.0, 3.0, 3.0, 3.0, 3.0]]> : tensor<2x5xf32> // U1

    %C = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}} 
        dense<[[2.0, 2.0, 2.0, 2.0, 0.0],  // L1
               [5.0, 5.0, 5.0, 5.0, 5.0]]> : tensor<2x5xf32> // M0

    %D = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}} 
        dense<[[10.0, 10.0, 10.0, 10.0, 10.0]]> : tensor<1x5xf32> // M0

    %c0 = arith.constant 0 : index
    %e0 = tensor.empty(%c0) : tensor<?x5xf32>
    %e1 = tensor.empty(%c0) : tensor<?x5xf32>
    %e2 = tensor.empty(%c0) : tensor<?x5xf32>

    %R1 = dia.elementwise kind = <add> ins(%A, %B : tensor<2x5xf32>, tensor<2x5xf32>) 
                                       outs(%e0 : tensor<?x5xf32>) -> tensor<?x5xf32>
                                       
    %R2 = dia.elementwise kind = <mul> ins(%R1, %C : tensor<?x5xf32>, tensor<2x5xf32>) 
                                       outs(%e1 : tensor<?x5xf32>) -> tensor<?x5xf32>

    %R3 = dia.elementwise kind = <sub> ins(%R2, %D : tensor<?x5xf32>, tensor<1x5xf32>) 
                                       outs(%e2 : tensor<?x5xf32>) -> tensor<?x5xf32>
                                       
    // Bufferize and Print
    %dim = tensor.dim %R3, %c0 : tensor<?x5xf32>
    %memref = memref.alloc(%dim) : memref<?x5xf32>
    bufferization.materialize_in_destination %R3 in writable %memref 
        : (tensor<?x5xf32>, memref<?x5xf32>) -> ()
    
    %cast = memref.cast %memref : memref<?x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<?x5xf32>
    
    return
  }
}
