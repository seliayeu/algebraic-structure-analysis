include "mlir/Pass/PassBase.td"

def AlgebraicStructureRewrite : Pass<"algebraic-structure-rewrite", "mlir::func::FuncOp"> {
    let summary = "Replace operations on diagonal and symmetric matrices with more efficient versions";
    let description = [{
        Run Algebraic Structure Analysis on a function to find operations that have operands deduced
        to be diagonal or symmetric matrices and rewrite them to use asymptotically faster versions.
    }];
}

def AlgebraicStructureDebug : Pass<"algebraic-structure-debug", "mlir::func::FuncOp"> {
    let summary = "Add properties deduced by the AlgebraicStructureAnalysis to attribute dictionaries";
    let description = [{
        Run Algebraic Structure Analysis on a function to deduce matrix properties and then add those
        matrix properties to their attribute dictionaries.
    }];
}
