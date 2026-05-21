func.func @kernel() -> f32{
    %cf1 = arith.constant 1.0 : f32
    %A_empty  = tensor.empty(): tensor<Yx1024xf32>
    %P0_empty = tensor.empty(): tensor<Yx1024xf32>
    %Q_empty  = tensor.empty(): tensor<Yx1024xf32>

    %A  = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%A_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
    %P0 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%P0_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
    %Q  = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%Q_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>

    %c0 = arith.constant 0 : index
    %e0 = tensor.empty() : tensor<1024x1024xf32>
    %e1 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e2 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e3 = tensor.empty(%c0) : tensor<?x1024xf32>

    %At = dia.transpose (%A: tensor<Yx1024xf32>)

    %R1 = dia.matmul ins(%A, %P0 : tensor<Yx1024xf32>, tensor<Yx1024xf32>)
                        outs(%e1 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
    %R2 = dia.matmul ins(%R1, %At: tensor<?x1024xf32>, tensor<Yx1024xf32>)
                        outs(%e2 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
    %P1 = dia.elementwise kind = <add> ins(%R2, %Q: tensor<?x1024xf32>, tensor<Yx1024xf32>) 
                                        outs(%e3 : tensor<?x1024xf32>) -> tensor<?x1024xf32>

    %zero = arith.constant 0: index
    %result = tensor.extract %P1[%zero, %zero]: tensor<?x1024xf32>
    return %result: f32
}
