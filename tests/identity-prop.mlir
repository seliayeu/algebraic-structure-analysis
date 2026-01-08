module {
    func.func @test_diagonal_prop() -> tensor<3x3xi32> {
        %0 = arith.constant { metadata = { analysisState = "diagonal" } } dense<[[2,0,0],[0,3,0],[0,0,7]]> : tensor<3x3xi32>
        %1 = arith.constant { metadata = { analysisState = "diagonal" } } dense<[[5,0,0],[0,4,0],[0,0,3]]> : tensor<3x3xi32>
        %3 = tensor.empty () : tensor<3x3xi32>
        %2 = linalg.matmul
            ins(%0, %1: tensor<3x3xi32>, tensor<3x3xi32>)
            outs(%3: tensor<3x3xi32>)
        -> tensor<3x3xi32>
        return %2: tensor<3x3xi32>
    }
}
