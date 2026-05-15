func.func @kernel() -> f32 {
  %Q  = arith.constant {metadata = {dia = true, lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}} dense<1.0>: tensor<2047x1024xf32>
  %K  = arith.constant {metadata = {dia = true, lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}} dense<1.0>: tensor<2047x1024xf32>
  %V  = arith.constant {metadata = {dia = true, lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}} dense<1.0>: tensor<2047x1024xf32>

  %mask  = arith.constant {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0>: tensor<Yx1024xf32>
  %factor = arith.constant {metadata = {dia = true, lowerBw = 1023 : i64, upperBw = 1023 : i64, propertyDims = [0, 1]}} dense<0.03125> : tensor<2047x1024xf32>

  %zero  = arith.constant 0.0 : f32
  %zidx = arith.constant 0 : index

  // K^t
  %K_t = dia.transpose (%K: tensor<2047x1024xf32>)

  // Q @ K^t
  %initZ = tensor.empty(%zidx) : tensor<?x1024xf32>

  %qkt = dia.matmul
      ins(%Q, %K_t: tensor<2047x1024xf32>, tensor<2047x1024xf32>)
      outs(%initZ : tensor<?x1024xf32>) -> tensor<?x1024xf32>


   // qkt (.) mask
   %maskedZ = tensor.empty(%zidx) : tensor<?x1024xf32>

   %masked = dia.elementwise kind = <mul> ins(%qkt, %mask: tensor<?x1024xf32>, tensor<Yx1024xf32>)
                                       outs(%maskedZ: tensor<?x1024xf32>) -> tensor<?x1024xf32>

   // scale masked values
   %scaledZ = tensor.empty(%zidx) : tensor<?x1024xf32>

   %scaled = dia.elementwise kind = <mul> ins(%masked, %factor: tensor<?x1024xf32>, tensor<2047x1024xf32>)
                                       outs(%scaledZ: tensor<?x1024xf32>) -> tensor<?x1024xf32>

   // softmax(scaled)
   %weights = dia.softmax(%scaled: tensor<?x1024xf32>) -> tensor<?x1024xf32>

   // weights @ V
   %outZ = tensor.empty(%zidx) : tensor<?x1024xf32>

   %out = dia.matmul
       ins(%weights, %V: tensor<?x1024xf32>, tensor<2047x1024xf32>)
       outs(%outZ: tensor<?x1024xf32>) -> tensor<?x1024xf32>

  %result = tensor.extract %out[%zidx, %zidx]: tensor<?x1024xf32>

  return %result: f32
}
