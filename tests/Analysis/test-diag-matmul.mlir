// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
    func.func @main() {
        %0 = arith.constant { metadata = { analysisState = "Diagonal", propertyDims = [0, 1] } } dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
        %1 = arith.constant { metadata = { analysisState = "Diagonal", propertyDims = [0, 1] } } dense<[[5.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,3.0]]> : tensor<3x3xf32>
        %3 = tensor.empty () : tensor<3x3xf32>

        // CHECK: analysisState = "Diagonal"
        // CHECK-SAME: propertyDims = [0, 1]
        %2 = linalg.matmul
            ins(%0, %1: tensor<3x3xf32>, tensor<3x3xf32>)
            outs(%3: tensor<3x3xf32>)
        -> tensor<3x3xf32>

        return
    }
}
