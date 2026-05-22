func.func @kernel() -> f32 {
  %cf1 = arith.constant 1.0 : f32
  %Q_empty  = tensor.empty(): tensor<1024x1024xf32>
  %K_empty  = tensor.empty(): tensor<1024x1024xf32>
  %V_empty  = tensor.empty(): tensor<1024x1024xf32>

  %Q  = linalg.fill {metadata = {lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%Q_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %K  = linalg.fill {metadata = {lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%K_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %V  = linalg.fill {metadata = {lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%V_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %mask_empty  = tensor.empty() : tensor<Yx1024xf32>
  %mask = linalg.fill {metadata = {dia = true, lowerBw = X: i64, upperBw = X: i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%mask_empty: tensor<Yx1024xf32>) -> tensor<Yx1024xf32>

  %factor_const = arith.constant 0.03125: f32
  %factor_empty = tensor.empty(): tensor<1024x1024xf32>
  %factor = linalg.fill {metadata = {lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}} ins(%factor_const: f32) outs(%factor_empty: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %zero  = arith.constant 0.0 : f32
  %zidx = arith.constant 0 : index

  // K^t
  // %K_t = dia.transpose (%K: tensor<1024x1024xf32>)

  %init_kT = tensor.empty() : tensor<1024x1024xf32>
  %K_t = linalg.transpose
    ins(%K : tensor<1024x1024xf32>)
    outs(%init_kT : tensor<1024x1024xf32>) permutation = [1, 0]



  // Q @ K^t
  %initZ = tensor.empty() : tensor<1024x1024xf32>

  %qkt = dia.matmul
      ins(%Q, %K_t: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%initZ : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


   // qkt (.) mask
   %maskedZ = tensor.empty(%zidx) : tensor<?x1024xf32>

   %masked = dia.elementwise kind = <mul> ins(%qkt, %mask: tensor<1024x1024xf32>, tensor<Yx1024xf32>)
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

  %result = tensor.extract %out[%zidx, %zidx]: tensor<?x1024xf32>

  return %result: f32
}
