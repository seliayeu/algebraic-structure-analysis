module {
  func.func @attention() -> f32 {
    %e_qk  = arith.constant dense<0.0> : tensor<ZxZxf32>
    %e_mqk = arith.constant dense<0.0> : tensor<ZxZxf32>
    %e_out = arith.constant dense<0.0> : tensor<ZxZxf32>

    %Q = arith.constant
        dense<1.0> : tensor<ZxZxf32>
    %K = arith.constant
        dense<1.0> : tensor<ZxZxf32>
    %V = arith.constant
        dense<1.0> : tensor<ZxZxf32>
    %Mask = arith.constant
        {metadata = {lowerBw = X : i64, upperBw = Y : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<ZxZxf32>

    %QK = linalg.matmul
        ins(%Q, %K  : tensor<ZxZxf32>, tensor<ZxZxf32>)
        outs(%e_qk  : tensor<ZxZxf32>) -> tensor<ZxZxf32>

    %MaskedQK = linalg.mul
        ins(%Mask, %QK : tensor<ZxZxf32>, tensor<ZxZxf32>)
        outs(%e_mqk    : tensor<ZxZxf32>) -> tensor<ZxZxf32>

    %AttnWeights = dia.softmax
        (%MaskedQK : tensor<ZxZxf32>) -> tensor<ZxZxf32>

    %Output = linalg.matmul
        ins(%AttnWeights, %V : tensor<ZxZxf32>, tensor<ZxZxf32>)
        outs(%e_out          : tensor<ZxZxf32>) -> tensor<ZxZxf32>

    %index  = arith.constant 0 : index
    %result = tensor.extract %Output[%index, %index] : tensor<ZxZxf32>
    return %result : f32
  }
}
