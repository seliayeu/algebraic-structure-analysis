#ifndef DIA_OPS_TD
#define DIA_OPS_TD

include "DIADialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class DIA_Op<string mnemonic, list<Trait> traits = []>
    : Op<DIA_Dialect, mnemonic, traits>;

def DIA_MatmulOp : DIA_Op<"matmul", [Pure]> {
    let summary = "DIA matrix multiplication with heterogeneous layouts";
    let arguments = (ins
        AnyRankedTensor:$lhs,
        AnyRankedTensor:$rhs,
        AnyRankedTensor:$output
    );
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
        `outs` `(` $output `:` type($output) `)`
        `->` type($result) attr-dict
    }];
}

#endif // DIA_OPS
