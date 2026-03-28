func.func @kernel() -> f32{
    %A  = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
    %P0 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
    %Q  = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>

    %c0 = arith.constant 0 : index
    %e0 = tensor.empty() : tensor<1024x1024xf32>
    %e1 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e2 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e3 = tensor.empty(%c0) : tensor<?x1024xf32>

    %At = linalg.transpose ins(%A  : tensor<1024x1024xf32>)
                           outs(%e0 : tensor<1024x1024xf32>)
                           permutation = [1, 0]

    %R1 = dia.matmul ins(%A, %P0 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                        outs(%e1 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
    %R2 = dia.matmul ins(%R1, %At: tensor<?x1024xf32>, tensor<1024x1024xf32>)
                        outs(%e2 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
    %P1 = dia.elementwise kind = <add> ins(%R2, %Q: tensor<?x1024xf32>, tensor<1024x1024xf32>) 
                                        outs(%e3 : tensor<?x1024xf32>) -> tensor<?x1024xf32>

    %zero = arith.constant 4: index
    %result = tensor.extract %P1[%zero, %zero]: tensor<?x1024xf32>
    return %result: f32
}
