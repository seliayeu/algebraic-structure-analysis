include "mlir/Pass/PassBase.td"

def BandedRewrite: Pass<"banded-rewrite", "mlir::func::FuncOp"> {
    let summary = "Replace operations on banded matrices with more efficient versions";
    let description = [{
        Run Banded Structure Analysis on a function to find operations that have operands deduced
        to have certain bandwidths to be replaced with cheaper versions
    }];
    let dependentDialects = [
      "tensor::TensorDialect",
    ];
}

def BandedAnalysis : Pass<"banded-analysis", "mlir::func::FuncOp"> {
    let summary = "Add properties deduced by the BandedStructureAnalysis to attribute dictionaries";
    let description = [{
        Run Banded Structure Analysis on a function to deduce matrix properties and then add those
        matrix properties to their attribute dictionaries.
    }];
    let options = [
      Option<"detectDIA",
            "detect-dia",
            "bool",
            /*default=*/"false",
            "Enable DIA compression">,
      Option<"heuristic",
            "heuristic",
            "bool",
            /*default=*/"false",
            "Enable from_dense insertion heuristic">
    ];
}
