#include "Transform/LinalgMatmulRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// -----------------------------------------------------------------------------
// Pattern
// -----------------------------------------------------------------------------

namespace mlir::bpa {
#define GEN_PASS_DEF_LINALGMATMULREWRITE
#include "lib/Transform/Passes.h.inc"

struct LinalgMatmulPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override {
        Location loc = op.getLoc();

        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value Cinit = op.getOutputs()[0];

        auto typeA = dyn_cast<RankedTensorType>(A.getType());
        auto typeB = dyn_cast<RankedTensorType>(B.getType());
        auto typeC = dyn_cast<RankedTensorType>(Cinit.getType());
        if (!typeA || !typeB || !typeC)
            return rewriter.notifyMatchFailure(
                op, "expected ranked tensor operands; already bufferized?");

        if (!typeA.getElementType().isF32())
            return rewriter.notifyMatchFailure(op, "only f32 element type supported");

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

        Value M = tensor::DimOp::create(rewriter, loc, A, 0);
        Value K = tensor::DimOp::create(rewriter, loc, A, 1);
        Value N = tensor::DimOp::create(rewriter, loc, B, 1);

        // -- i loop ---------------------------------------------------------------
        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, M, c1, /*iterArgs=*/ValueRange{ Cinit },
            [&](OpBuilder& b, Location loc, Value i, ValueRange iArgs) {
                Value Ci = iArgs[0];

                // -- k loop ---------------------------------------------------------
                auto kLoop = scf::ForOp::create(
                    b, loc, c0, K, c1, ValueRange{ Ci },
                    [&](OpBuilder& b, Location loc, Value k, ValueRange kArgs) {
                        Value Ck = kArgs[0];

                        // A[i,k] invariant w.r.t. j — extract once, lives in a
                        // register across the entire inner j loop
                        Value a = tensor::ExtractOp::create(b, loc, A, ValueRange{ i, k });

                        // -- j loop (innermost) ---------------------------------------
                        auto jLoop = scf::ForOp::create(
                            b, loc, c0, N, c1, ValueRange{ Ck },
                            [&](OpBuilder& b, Location loc, Value j, ValueRange jArgs) {
                                Value Cj = jArgs[0];

                                // B[k,j]: j strides along a row -> sequential
                                Value bv = tensor::ExtractOp::create(b, loc, B, ValueRange{ k, j });
                                // C[i,j]: j strides along a row -> sequential
                                Value cv =
                                    tensor::ExtractOp::create(b, loc, Cj, ValueRange{ i, j });

                                Value mul = arith::MulFOp::create(b, loc, a, bv);
                                Value acc = arith::AddFOp::create(b, loc, cv, mul);
                                Value Cnew =
                                    tensor::InsertOp::create(b, loc, acc, Cj, ValueRange{ i, j });

                                scf::YieldOp::create(b, loc, Cnew);
                            });

                        scf::YieldOp::create(b, loc, jLoop.getResult(0));
                    });

                scf::YieldOp::create(b, loc, kLoop.getResult(0));
            });

        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }
};

// -----------------------------------------------------------------------------
// Pass
// -----------------------------------------------------------------------------

struct LinalgMatmulRewrite : public impl::LinalgMatmulRewriteBase<LinalgMatmulRewrite> {
    using LinalgMatmulRewriteBase::LinalgMatmulRewriteBase;

    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = funcOp.getContext();

        RewritePatternSet patterns(context);
        patterns.add<LinalgMatmulPattern>(context);

        GreedyRewriteConfig config;
        config.setMaxIterations(1);
        config.setUseTopDownTraversal(true);

        (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
    }
};

}  // namespace mlir::bpa
