#include <cassert>
#include "lib/Transforms/AlgebraicStructureRewrite.h"
#include "lib/Analysis/AlegbraicStructureAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::asa {

#define GEN_PASS_DEF_ALGEBRAICSTRUCTUREREWRITE
#include "lib/Transforms/AlgebraicStructureRewrite.h.inc"

struct AddRewrite : public OpRewritePattern<linalg::AddOp> {
    AddRewrite(MLIRContext* context, DataFlowSolver* solver) : OpRewritePattern<linalg::AddOp>(context), solver{ solver } {};

    LogicalResult matchAndRewrite(linalg::AddOp addOp, PatternRewriter& rewriter) const override {
        auto operands{ addOp.getInputs() };
        auto lhs{ operands[0] };
        auto rhs{ operands[1] };

        auto lhsState{  solver->lookupState<dataflow::Lattice<AlgebraicStructureAnalysisLatticeValue>, Value>(lhs) };
        auto rhsState{  solver->lookupState<dataflow::Lattice<AlgebraicStructureAnalysisLatticeValue>, Value>(rhs) };
       
        if (!lhsState || !rhsState || !(lhsState->getValue() == rhsState->getValue()))
            return success();

        auto lhsProperty{ lhsState->getValue().getState() };
        auto rhsProperty{ rhsState->getValue().getState() };

        if (lhsProperty != AlgebraicStructureAnalysisLatticeValue::AlgebraicProperty::Symmetric)
            return success();

        // rewrite the output matrix symmetricly
        auto type{ dyn_cast<TensorType>(lhs.getType()) };
        auto shape{ type.getShape() };

        assert(shape.size() == 2 && shape[0] == shape[1]);

        auto n{ shape[0] };

        auto lowerBound{ arith::ConstantIndexOp::create(rewriter, addOp.getLoc(), 0) };
        auto upperBound{ arith::ConstantIndexOp::create(rewriter, addOp.getLoc(), n) };
        auto step{ arith::ConstantIndexOp::create(rewriter, addOp.getLoc(), 1) };

        auto emptyTensorOp{ tensor::EmptyOp::create(rewriter, addOp.getLoc(), shape, type.getElementType()) };
        auto outerForOp{ scf::ForOp::create(rewriter, addOp.getLoc(), lowerBound, upperBound, step, { emptyTensorOp }, 
            [&](OpBuilder& b, Location loc, Value ivOuter, ValueRange iterArgsOuter) {
                auto currentMatrix{ iterArgsOuter[0] };
                auto oneIndexOp{ arith::ConstantIndexOp::create(rewriter, addOp.getLoc(), 1) };
                auto ivOuterPlusOne{ arith::AddIOp::create(rewriter, addOp.getLoc(), ivOuter, oneIndexOp)};
                auto innerForOp{ scf::ForOp::create(rewriter, addOp.getLoc(), lowerBound, ivOuterPlusOne, step, currentMatrix,
                    [&](OpBuilder& b, Location loc, Value ivInner, ValueRange iterArgsInner) {
                        auto currentMatrix{ iterArgsInner[0] };
                        auto lhsExtractOp{ tensor::ExtractOp::create(rewriter, addOp.getLoc(), lhs, {ivInner, ivOuter}) };
                        auto rhsExtractOp{ tensor::ExtractOp::create(rewriter, addOp.getLoc(), rhs, {ivInner, ivOuter}) };
                        auto innerAddOp{ arith::AddFOp::create(rewriter, addOp.getLoc(), lhsExtractOp, rhsExtractOp) };
                        auto insertOpLower{ tensor::InsertOp::create(rewriter, addOp.getLoc(), type, innerAddOp, currentMatrix, {ivInner, ivOuter}) };
                        auto insertOpUpper{ tensor::InsertOp::create(rewriter, addOp.getLoc(), type, innerAddOp, insertOpLower, {ivOuter, ivInner}) };
                        scf::YieldOp::create(rewriter, addOp.getLoc(), { insertOpUpper });
                    }
                )};
                scf::YieldOp::create(rewriter, addOp.getLoc(), { innerForOp.getResult(0) });
            } 
        )};

        rewriter.replaceOp(addOp, outerForOp);
        
        return success();
    }
private:
    DataFlowSolver* solver;
};

struct MatmulRewrite : public OpRewritePattern<linalg::MatmulOp> {
    MatmulRewrite(MLIRContext* context, DataFlowSolver* solver) : OpRewritePattern<linalg::MatmulOp>(context), solver{ solver } {};

    LogicalResult rewriteDiagonalTimesDiagonal(linalg::MatmulOp matmulOp, PatternRewriter& rewriter) const {
        auto lhs{ matmulOp.getInputs()[0] };
        auto rhs{ matmulOp.getInputs()[1] };

        auto type{ dyn_cast<TensorType>(lhs.getType()) };
        auto shape{ type.getShape() };

        assert(shape.size() == 2 && shape[0] == shape[1]);

        auto n{ shape[0] };

        auto lowerBound{ arith::ConstantIndexOp::create(rewriter, matmulOp.getLoc(), 0) };
        auto upperBound{ arith::ConstantIndexOp::create(rewriter, matmulOp.getLoc(), n) };
        auto step{ arith::ConstantIndexOp::create(rewriter, matmulOp.getLoc(), 1) };

        auto emptyTensorOp{ tensor::EmptyOp::create(rewriter, matmulOp.getLoc(), shape, type.getElementType()) };
        auto outerForOp{ scf::ForOp::create(rewriter, matmulOp.getLoc(), lowerBound, upperBound, step, { emptyTensorOp }, 
            [&](OpBuilder& b, Location loc, Value iv, ValueRange iterArgsOuter) {
                auto currentMatrix{ iterArgsOuter[0] };
                auto lhsExtractOp{ tensor::ExtractOp::create(rewriter, matmulOp.getLoc(), {iv, iv}) };
                auto rhsExtractOp{ tensor::ExtractOp::create(rewriter, matmulOp.getLoc(), {iv, iv}) };
                auto addOp{ arith::AddFOp::create(rewriter, matmulOp.getLoc(), lhsExtractOp, rhsExtractOp) };
                auto insertOp{ tensor::InsertOp::create(rewriter, matmulOp.getLoc(), type, addOp, currentMatrix, {iv, iv}) };
                scf::YieldOp::create(rewriter, addOp.getLoc(), { insertOp });
            } 
        )};
        rewriter.replaceOp(matmulOp, outerForOp);

        return success();       
    }

    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp, PatternRewriter& rewriter) const override {
        auto operands{ matmulOp.getInputs() };
        auto lhs{ operands[0] };
        auto rhs{ operands[1] };

        auto lhsState{  solver->lookupState<dataflow::Lattice<AlgebraicStructureAnalysisLatticeValue>, Value>(lhs) };
        auto rhsState{  solver->lookupState<dataflow::Lattice<AlgebraicStructureAnalysisLatticeValue>, Value>(rhs) };
       
        if (!lhsState || !rhsState)
            return success();

        auto lhsProperty{ lhsState->getValue().getState() };
        auto rhsProperty{ rhsState->getValue().getState() };

        if (lhsProperty == AlgebraicStructureAnalysisLatticeValue::AlgebraicProperty::Diagonal &&
                rhsProperty == AlgebraicStructureAnalysisLatticeValue::AlgebraicProperty::Diagonal)
            return rewriteDiagonalTimesDiagonal(matmulOp, rewriter);

        return success();
    }
private:
    DataFlowSolver* solver;
};

struct AlgebraicStructureRewritePass : public impl::AlgebraicStructureRewriteBase<AlgebraicStructureRewritePass> {
    using AlgebraicStructureRewriteBase::AlgebraicStructureRewriteBase;
    void runOnOperation() {
        auto* op{ getOperation() };       
        DataFlowSolver solver{};
        solver.load<dataflow::DeadCodeAnalysis>();
        solver.load<AlgebraicStructureAnalysis>();
        if (failed(solver.initializeAndRun(op)))
            return signalPassFailure();
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<AddRewrite>(&getContext(), &solver);
        (void)applyPatternsGreedily(getOperation(), std::move(patterns));

        return;
    }
};

}
