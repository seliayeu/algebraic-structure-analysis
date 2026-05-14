// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
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
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format,%llvm_root/build/lib/libmlir_c_runner_utils.%lib_format> %t
// RUN: FileCheck %s < %t

func.func private @printMemrefF32(memref<*xf32>)
func.func @main() {
  %Q  = arith.constant {metadata = {dia = true, lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}} dense<1.0>: tensor<7x4xf32>
  %K  = arith.constant {metadata = {dia = true, lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}} dense<1.1>: tensor<7x4xf32>
  %V  = arith.constant {metadata = {dia = true, lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}} dense<1.2>: tensor<7x4xf32>

  %mask  = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1x4xf32>
  %factor = arith.constant {metadata = {dia = true, lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}} dense<0.03125>: tensor<7x4xf32>

  %zidx = arith.constant 0 : index

  // K^t
  %K_t = dia.transpose (%K: tensor<7x4xf32>)

  // Q @ K^t
  %initZ = tensor.empty(%zidx) : tensor<?x4xf32>

  %qkt = dia.matmul
      ins(%Q, %K_t: tensor<7x4xf32>, tensor<7x4xf32>)
      outs(%initZ : tensor<?x4xf32>) -> tensor<?x4xf32>


   // qkt (.) mask
   %maskedZ = tensor.empty(%zidx) : tensor<?x4xf32>

   %masked = dia.elementwise kind = <mul> ins(%qkt, %mask: tensor<?x4xf32>, tensor<1x4xf32>)
                                       outs(%maskedZ: tensor<?x4xf32>) -> tensor<?x4xf32>

   // scale masked values
   %scaledZ = tensor.empty(%zidx) : tensor<?x4xf32>

   %scaled = dia.elementwise kind = <mul> ins(%masked, %factor: tensor<?x4xf32>, tensor<7x4xf32>)
                                       outs(%scaledZ: tensor<?x4xf32>) -> tensor<?x4xf32>

   // softmax(scaled)
   %weights = dia.softmax(%scaled: tensor<?x4xf32>) -> tensor<?x4xf32>

   // weights @ V
   %outZ = tensor.empty(%zidx) : tensor<?x4xf32>

   %out = dia.matmul
       ins(%weights, %V: tensor<?x4xf32>, tensor<7x4xf32>)
       outs(%outZ: tensor<?x4xf32>) -> tensor<?x4xf32>


   %dim = tensor.dim %out, %zidx: tensor<?x4xf32>
   %memref = memref.alloc(%dim) : memref<?x4xf32>
   bufferization.materialize_in_destination %out in writable %memref
       : (tensor<?x4xf32>, memref<?x4xf32>) -> ()
   %cast = memref.cast %memref : memref<?x4xf32> to memref<*xf32>
   call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
   memref.dealloc %memref : memref<?x4xf32>
   // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [7, 4] strides = [4, 1] data =
   // CHECK-NEXT: {{\[\[}}1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
   // CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
   // CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
   // CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
   // CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
   // CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2],
   // CHECK-NEXT:  [1.2{{.*}}1.2{{.*}}1.2{{.*}}1.2]]

  return
}
