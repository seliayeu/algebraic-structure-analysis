// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.so > %t
// RUN: FileCheck %s < %t

// tridiag(1.0, lowerBw=1, upperBw=1) * tridiag(2.0, lowerBw=1, upperBw=1)
// result: DIA with lowerBw=2, upperBw=2 -> 5x10
// CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [5, 10]
// CHECK: {{\[}}[0{{.*}}, 0{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}],
// CHECK-NEXT: [0{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}],
// CHECK-NEXT: [4{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 4{{.*}}],
// CHECK-NEXT: [4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 0{{.*}}],
// CHECK-NEXT: [2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 0{{.*}}, 0{{.*}}]]
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %I0 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<10x10xf32>
    %I1 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<2.0> : tensor<10x10xf32>
    %e1 = arith.constant dense<0.0> : tensor<5x10xf32>
    %R0 = dia.matmul ins(%I0, %I1 : tensor<10x10xf32>, tensor<10x10xf32>)
                     outs(%e1 : tensor<5x10xf32>) -> tensor<5x10xf32>
    %memref = memref.alloc() : memref<5x10xf32>
    bufferization.materialize_in_destination %R0 in %memref {writable}
        : (tensor<5x10xf32>, memref<5x10xf32>) -> ()
    %cast = memref.cast %memref : memref<5x10xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<5x10xf32>
    return
  }
  
}
