include "mlir/Pass/PassBase.td"

def BandedStructureRewrite : Pass<"banded-structure-rewrite", "mlir::func::FuncOp"> {
    let summary = "Replace operations on banded matrices with more efficient versions";
    let description = [{
        Run Banded Structure Analysis on a function to find operations that have operands deduced
        to have certain bandwidths to be replaced with cheaper versions
    }];
}

def BandedStructureDebug : Pass<"banded-structure-debug", "mlir::func::FuncOp"> {
    let summary = "Add properties deduced by the BandedStructureAnalysis to attribute dictionaries";
    let description = [{
        Run Banded Structure Analysis on a function to deduce matrix properties and then add those
        matrix properties to their attribute dictionaries.
    }];
}
