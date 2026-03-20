// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format> %t
// RUN: FileCheck %s < %t

module {
  func.func private @printMemrefF32(memref<*xf32>)


  func.func @main() {

    %t0 = arith.constant dense<0.0>: tensor<4x4xf32>
    %t1 = arith.constant dense<0.0>: tensor<4x4xf32>
    %t2 = arith.constant dense<0.0>: tensor<4x4xf32>
    %t3 = arith.constant dense<0.0>: tensor<4x4xf32>
    %t4 = arith.constant dense<0.0>: tensor<4x4xf32>

    // diagonal
    %i0 = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[1.0, 0.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0],
                   [0.0, 0.0, 3.0, 0.0],
                   [0.0, 0.0, 0.0, 4.0]]> : tensor<4x4xf32>


    // diagonal
    %i1 = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[1.0, 0.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0, 0.0],
                   [0.0, 0.0, 3.0, 0.0],
                   [0.0, 0.0, 0.0, 4.0]]> : tensor<4x4xf32>

    // tridiagonal
    %i2 = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64,  propertyDims = [0, 1]}}
            dense<[[1.0, 1.0, 0.0, 0.0],
                   [1.0, 2.0, 2.0, 0.0],
                   [0.0, 2.0, 3.0, 3.0],
                   [0.0, 0.0, 3.0, 4.0]]> : tensor<4x4xf32>


    // upper-triangle
    %i3 = arith.constant {metadata = {upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
            dense<[[1.0, 1.0, 1.0, 1.0],
                   [0.0, 2.0, 2.0, 2.0],
                   [0.0, 0.0, 3.0, 3.0],
                   [0.0, 0.0, 0.0, 4.0]]> : tensor<4x4xf32>

    // i0 * i1
    // diag * diag
    %r0 = linalg.matmul ins(%i0, %i1 : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%t0 : tensor<4x4xf32>) -> tensor<4x4xf32>


    // r0 * i2
    // diag * tridiagonal
    %r1 = linalg.matmul ins(%r0, %i2 : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%t1 : tensor<4x4xf32>) -> tensor<4x4xf32>



    // r1 * i3
    // tridiagonal * upper-triangle
    %r2 = linalg.matmul ins(%r1, %i3 : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%t2 : tensor<4x4xf32>) -> tensor<4x4xf32>

    // r1 + r2
    // tridiagonal + Band(1, 2)
    %r3 = linalg.add ins(%r1, %r2: tensor<4x4xf32>, tensor<4x4xf32>) 
                      outs(%t3: tensor<4x4xf32>) -> tensor<4x4xf32>

    // call @print(%r3): (tensor<4x4xf32>) -> ()
    // r3 . i0
    // Band(1, 2) . diag
    %r4 = linalg.mul ins(%r3, %i0: tensor<4x4xf32>, tensor<4x4xf32>) 
                      outs(%t4: tensor<4x4xf32>) -> tensor<4x4xf32>

    // The line below should print a matrix with diagonal [2, 56, 432, 1856]
    // CHECK: {{\[}}[2
    // CHECK-NEXT: [0{{.*}}, 56
    // CHECK-NEXT: [0{{.*}}, 0{{.*}}, 432
    // CHECK-NEXT: [0{{.*}}, 0{{.*}}, 0{{.*}}, 1856
    %memref = memref.alloc() : memref<4x4xf32>
    bufferization.materialize_in_destination %r4 in %memref {writable}
        : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
    %cast = memref.cast %memref : memref<4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<4x4xf32>

    return
  }
}
