// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4]
// CHECK: {{\[}}[1{{.*}}, 0{{.*}}, 0{{.*}}, 0{{.*}}],
// CHECK-NEXT: [1{{.*}}, 2{{.*}}, 0{{.*}}, 0{{.*}}],
// CHECK-NEXT: [1{{.*}}, 2{{.*}}, 3{{.*}}, 0{{.*}}],
// CHECK-NEXT: [1{{.*}}, 2{{.*}}, 3{{.*}}, 4{{.*}}]]

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %0 = arith.constant {metadata = {upperBw = 3 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 1.0, 1.0],
               [0.0, 2.0, 2.0, 2.0],
               [0.0, 0.0, 3.0, 3.0],
               [0.0, 0.0, 0.0, 4.0]]> : tensor<4x4xf32>
    %1 = tensor.empty() : tensor<4x4xf32>
    %2 = linalg.transpose
        ins(%0 : tensor<4x4xf32>)
        outs(%1 : tensor<4x4xf32>)
        permutation = [1, 0]
    %memref = memref.alloc() : memref<4x4xf32>
    bufferization.materialize_in_destination %2 in %memref {writable}
        : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
    %cast = memref.cast %memref : memref<4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<4x4xf32>
    return
  }
}
