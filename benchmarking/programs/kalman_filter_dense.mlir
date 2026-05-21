func.func @kernel() -> f32{
    %cf1 = arith.constant 1.0 : f32
    %A_empty  = tensor.empty(): tensor<1024x1024xf32>
    %P0_empty = tensor.empty(): tensor<1024x1024xf32>
    %Q_empty  = tensor.empty(): tensor<1024x1024xf32>

    %A  = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%A_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %P0 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%P0_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %Q  = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%Q_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


    %e0 = tensor.empty(): tensor<1024x1024xf32>
    %e1 = tensor.empty(): tensor<1024x1024xf32>
    %e2 = tensor.empty(): tensor<1024x1024xf32>
    %e3 = tensor.empty(): tensor<1024x1024xf32>

    %At = linalg.transpose ins(%A  : tensor<1024x1024xf32>)
                           outs(%e0 : tensor<1024x1024xf32>)
                           permutation = [1, 0]

    %R1 = linalg.matmul ins(%A, %P0 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                        outs(%e1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %R2 = linalg.matmul ins(%R1, %At: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                        outs(%e2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %P1 = linalg.add ins(%R2, %Q : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%e3 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

    %zero = arith.constant 0: index
    %result = tensor.extract %P1[%zero, %zero]: tensor<1024x1024xf32>
    return %result: f32
}
