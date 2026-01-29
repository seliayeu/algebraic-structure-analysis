// RUN: ../build/tools/alg-opt %s \
// RUN:  --algebraic-structure-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=../../llvm-project/build/lib/libmlir_runner_utils.so | \
// RUN: FileCheck %s

module {
    func.func private @printMemrefF32(memref<*xf32>)
    func.func @main() {
        %0 = arith.constant { metadata = { analysisState = "Symmetric" } } dense<[[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
        %1 = arith.constant { metadata = { analysisState = "Symmetric" } } dense<[[5.0,8.0,2.0],[8.0,4.0,3.0],[2.0,3.0,3.0]]> : tensor<3x3xf32>
        %3 = tensor.empty() : tensor<3x3xf32>
        
        %2 = linalg.add
            ins(%0, %1 : tensor<3x3xf32>, tensor<3x3xf32>)
            outs(%3 : tensor<3x3xf32>)
            -> tensor<3x3xf32>

        %memref = memref.alloc() : memref<3x3xf32>
        %output = memref.alloc() : memref<3x3xf32>
        bufferization.materialize_in_destination %2 in %memref {writable} : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
        %cast = memref.cast %memref : memref<3x3xf32> to memref<*xf32>

        // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1]
        // CHECK-NEXT: [7,   12,   7]
        // CHECK-NEXT: [12,   7,   12]
        // CHECK-NEXT: [7,   12,   10]

        call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
        memref.dealloc %memref : memref<3x3xf32>
        return
    }
}
