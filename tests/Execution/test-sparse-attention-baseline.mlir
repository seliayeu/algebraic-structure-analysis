// RUN: %build/tools/alg-opt %s --dense-softmax-rewrite \
// RUN: --empty-tensor-to-alloc-tensor \
// RUN: --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN: --convert-linalg-to-loops \
// RUN: --convert-scf-to-cf \
// RUN: --convert-math-to-llvm \
// RUN: --convert-math-to-libm \
// RUN: --expand-strided-metadata \
// RUN: --lower-affine \
// RUN: --convert-arith-to-llvm \
// RUN: --convert-func-to-llvm \
// RUN: --convert-cf-to-llvm \
// RUN: --finalize-memref-to-llvm \
// RUN: --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format> %t
// RUN: FileCheck %s < %t

func.func private @printMemrefF32(memref<*xf32>)
func.func @main() {
  %Q  = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}}  dense<1.0> : tensor<4x4xf32>
  %K  = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}}  dense<1.1> : tensor<4x4xf32>
  %V  = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}}  dense<1.2> : tensor<4x4xf32>

  %mask  = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>

  %factor = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}} dense<0.03125> : tensor<4x4xf32>

  %zero  = arith.constant 0.0 : f32

  // K^t
  %init_kT = tensor.empty() : tensor<4x4xf32>
  %K_t = linalg.transpose
    ins(%K : tensor<4x4xf32>)
    outs(%init_kT : tensor<4x4xf32>) permutation = [1, 0]


  // Q @ K^t
  %init  = tensor.empty() : tensor<4x4xf32>
  %initZ = linalg.fill ins(%zero : f32)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>

  %qkt = linalg.matmul
      ins(%Q, %K_t: tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%initZ : tensor<4x4xf32>) -> tensor<4x4xf32>


  // qkt (.) mask
  %masked_init = tensor.empty() : tensor<4x4xf32>
  %maskedZ = linalg.fill ins(%zero : f32) outs(%masked_init: tensor<4x4xf32>) -> tensor<4x4xf32>

  %masked = linalg.elementwise kind=#linalg.elementwise_kind<mul>
      ins(%qkt, %mask : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%maskedZ : tensor<4x4xf32>) -> tensor<4x4xf32>

  // scale masked values
  %scaled_init = tensor.empty() : tensor<4x4xf32>
  %scaledZ = linalg.fill ins(%zero : f32) outs(%scaled_init: tensor<4x4xf32>) -> tensor<4x4xf32>

  %scaled = linalg.elementwise kind=#linalg.elementwise_kind<mul>
      ins(%masked, %factor: tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%scaledZ: tensor<4x4xf32>) -> tensor<4x4xf32>


  // softmax(scaled)
  %weights = dia.softmax(%scaled: tensor<4x4xf32>) -> tensor<4x4xf32>

  // weights @ V
  %out_init = tensor.empty() : tensor<4x4xf32>
  %outZ = linalg.fill ins(%zero : f32)
                       outs(%out_init: tensor<4x4xf32>) -> tensor<4x4xf32>
  %out = linalg.matmul
    ins(%weights, %V : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%outZ: tensor<4x4xf32>) -> tensor<4x4xf32>

  %memref = memref.alloc() : memref<4x4xf32>
  bufferization.materialize_in_destination %out in %memref {writable}
      : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
  %cast = memref.cast %memref : memref<4x4xf32> to memref<*xf32>
  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  memref.dealloc %memref : memref<4x4xf32>

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
// CHECK-NEXT: {{\[\[}}1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
// CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
// CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
// CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2]]
  return
}
