include "mlir/Pass/PassBase.td"

def AlgebraicStructureRewrite : Pass<"algebraic-structure-rewrite"> {
    let summary = "Replace operations on diagonal and symmetric matrices with more efficient versions";
    let description = [{
        Run Algebraic Structure Analysis on a function to find operations that have operands deduced
        to be diagonal or symmetric matrices and rewrite them to use asymptotically faster versions.
    }];
}
