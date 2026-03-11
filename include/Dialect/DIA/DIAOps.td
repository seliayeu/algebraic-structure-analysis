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
        2DTensorOf<[AnyType]>:$lhs,
        2DTensorOf<[AnyType]>:$rhs,
        2DTensorOf<[AnyType]>:$output
    );
    let results = (outs 2DTensorOf<[AnyType]>:$result);
    let assemblyFormat = [{
        `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
        `outs` `(` $output `:` type($output) `)`
        `->` type($result) attr-dict
    }];
}

def DIA_BatchMatmulOp : DIA_Op<"batch_matmul", [Pure]> {
    let summary = "DIA batched matrix multiplication with heterogeneous layouts";
    let arguments = (ins
        3DTensorOf<[AnyType]>:$lhs,
        3DTensorOf<[AnyType]>:$rhs,
        3DTensorOf<[AnyType]>:$output
    );
    let results = (outs 3DTensorOf<[AnyType]>:$result);
    let assemblyFormat = [{
        `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
        `outs` `(` $output `:` type($output) `)`
        `->` type($result) attr-dict
    }];
}

def DIA_FromDenseOp : DIA_Op<"from_dense", [Pure]> {
    let summary = "Construct a DIA tensor from a dense constant tensor";

    let arguments = (ins
      AnyTensor:$input
    );

    let results = (outs
      AnyTensor:$output
    );

    let assemblyFormat = [{
      $input attr-dict `:` type($input) `->` type($output)
    }];

    let hasCanonicalizer = 1;
}


#endif // DIA_OPS
