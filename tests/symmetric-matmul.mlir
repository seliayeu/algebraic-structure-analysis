module {
    func.func @test_diagonal_prop() -> tensor<3x3xi32> {
        %0 = arith.constant { metadata = { analysisState = "Symmetric" } } dense<[[2,4,5],[4,3,9],[5,9,7]]> : tensor<3x3xi32>
        %1 = arith.constant { metadata = { analysisState = "Symmetric" } } dense<[[5,8,2],[8,4,3],[2,3,3]]> : tensor<3x3xi32>
        %3 = tensor.empty () : tensor<3x3xi32>
        %2 = linalg.matmul
            ins(%0, %1: tensor<3x3xi32>, tensor<3x3xi32>)
            outs(%3: tensor<3x3xi32>)
        -> tensor<3x3xi32>
        return %2: tensor<3x3xi32>
    }
}
