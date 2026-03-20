// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.so > %t
// RUN: FileCheck %s < %t

// diag(1.0) * diag(2.0) = diag(2.0)
// CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [1, 10]
// CHECK: {{\[}}[2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}]]
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %I0 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<10x10xf32>
    %I1 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<2.0> : tensor<10x10xf32>
    %e0 = arith.constant dense<0.0> : tensor<1x10xf32>
    %R1 = dia.matmul ins(%I0, %I1 : tensor<10x10xf32>, tensor<10x10xf32>)
                     outs(%e0 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %memref = memref.alloc() : memref<1x10xf32>
    bufferization.materialize_in_destination %R1 in %memref {writable}
        : (tensor<1x10xf32>, memref<1x10xf32>) -> ()
    %cast = memref.cast %memref : memref<1x10xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<1x10xf32>
    return
  }
}
