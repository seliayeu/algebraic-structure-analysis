func.func @kernel() -> f32 {
  %Q  = tensor.empty() : tensor<1024x1024xf32>
  %K  = tensor.empty() : tensor<1024x1024xf32>
  %V  = tensor.empty() : tensor<1024x1024xf32>

  %mask  = tensor.empty() : tensor<1024x1024xf32>
  %factor = arith.constant dense<0.03125> : tensor<1024x1024xf32>

  %zero  = arith.constant 0.0 : f32

  // K^t
  %init_kT = tensor.empty() : tensor<1024x1024xf32>
  %K_t = linalg.transpose
    ins(%K : tensor<1024x1024xf32>)
    outs(%init_kT : tensor<1024x1024xf32>) permutation = [1, 0]


  // Q @ K^t
  %init  = tensor.empty() : tensor<1024x1024xf32>
  %initZ = linalg.fill ins(%zero : f32)
                       outs(%init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %qkt = linalg.matmul
      ins(%Q, %K_t: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%initZ : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  // qkt (.) mask
  %masked_init = tensor.empty() : tensor<1024x1024xf32>
  %maskedZ = linalg.fill ins(%zero : f32) outs(%masked_init: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %masked = linalg.elementwise kind=#linalg.elementwise_kind<mul>
      ins(%qkt, %mask : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%maskedZ : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  // scale masked values
  %scaled_init = tensor.empty() : tensor<1024x1024xf32>
  %scaledZ = linalg.fill ins(%zero : f32) outs(%scaled_init: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %scaled = linalg.elementwise kind=#linalg.elementwise_kind<mul>
      ins(%masked, %factor: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%scaledZ: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  // softmax(scaled)
  %weights = dia.softmax(%scaled: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  // weights @ V
  %out_init = tensor.empty() : tensor<1024x1024xf32>
  %outZ = linalg.fill ins(%zero : f32)
                       outs(%out_init: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %out = linalg.matmul
    ins(%weights, %V : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
    outs(%outZ: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %zidx = arith.constant 0: index
  %result = tensor.extract %out[%zidx, %zidx]: tensor<1024x1024xf32>

  return %result: f32
}
