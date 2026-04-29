func.func @kernel() -> f32 {
  %Q  = tensor.empty() {metadata = {lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}}: tensor<1024x1024xf32>
  %K  = tensor.empty() {metadata = {lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}}: tensor<1024x1024xf32>
  %V  = tensor.empty() {metadata = {lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}}: tensor<1024x1024xf32>

  %mask  = tensor.empty() {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} : tensor<1024x1024xf32>
  %factor = arith.constant dense<0.03125> : tensor<1024x1024xf32>

  %zero  = arith.constant 0.0 : f32
  %zidx = arith.constant 0 : index

  // K^t
  %init_kT = tensor.empty() : tensor<1024x1024xf32>
  %K_t = linalg.transpose
    ins(%K : tensor<1024x1024xf32>)
    outs(%init_kT : tensor<1024x1024xf32>) permutation = [1, 0]


  // Q @ K^t
  %initZ = tensor.empty(%zidx) : tensor<?x1024xf32>

  %qkt = dia.matmul
      ins(%Q, %K_t: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%initZ : tensor<?x1024xf32>) -> tensor<?x1024xf32>


  // qkt (.) mask
  %maskedZ = tensor.empty(%zidx) : tensor<?x1024xf32>

  %masked = dia.elementwise kind = <mul> ins(%qkt, %mask: tensor<?x1024xf32>, tensor<1024x1024xf32>)
                                      outs(%maskedZ: tensor<?x1024xf32>) -> tensor<?x1024xf32>

  // scale masked values
  %scaledZ = tensor.empty(%zidx) : tensor<?x1024xf32>

  %scaled = dia.elementwise kind = <mul> ins(%masked, %factor: tensor<?x1024xf32>, tensor<1024x1024xf32>)
                                      outs(%scaledZ: tensor<?x1024xf32>) -> tensor<?x1024xf32>

  // softmax(scaled)
  %weights = dia.softmax(%scaled: tensor<?x1024xf32>) -> tensor<?x1024xf32>

  // weights @ V
  %outZ = tensor.empty(%zidx) : tensor<?x1024xf32>

  %out = dia.matmul
      ins(%weights, %V: tensor<?x1024xf32>, tensor<1024x1024xf32>)
      outs(%outZ: tensor<?x1024xf32>) -> tensor<?x1024xf32>


  // %dim = tensor.dim %out, %zidx: tensor<?x1024xf32>
  // %memref = memref.alloc(%dim) : memref<?x1024xf32>
  // bufferization.materialize_in_destination %out in writable %memref
  //     : (tensor<?x1024xf32>, memref<?x1024xf32>) -> ()
  // %cast = memref.cast %memref : memref<?x1024xf32> to memref<*xf32>
  // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  // memref.dealloc %memref : memref<?x1024xf32>
  %result = tensor.extract %out[%zidx, %zidx]: tensor<?x1024xf32>

  return %result: f32
}
