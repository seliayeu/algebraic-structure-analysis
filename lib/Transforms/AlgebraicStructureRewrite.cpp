#include "lib/Transforms/AlgebraicStructureRewrite.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::asa {

#define GEN_PASS_DEF_ALGEBRAICSTRUCTUREREWRITE
#include "lib/Transforms/AlgebraicStructureRewrite.h.inc"

struct AlgebraicStructureRewrite : public impl::AlgebraicStructureRewriteBase<AlgebraicStructureRewrite> {
    using AlgebraicStructureRewriteBase::AlgebraicStructureRewriteBase;
    void runOnOperation() {
        
    }
};

}
