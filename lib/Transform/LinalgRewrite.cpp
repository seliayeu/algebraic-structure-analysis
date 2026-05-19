#include "Transform/LinalgRewrite.h"

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
#define GEN_PASS_DEF_LINALGREWRITE
#include "lib/Transform/Passes.h.inc"

struct LinalgBatchMatmulPattern : public OpRewritePattern<linalg::BatchMatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                  PatternRewriter& rewriter) const override {
        Location loc = op.getLoc();

        Value A = op.getInputs()[0];       // [batch x M x K]
        Value B = op.getInputs()[1];       // [batch x K x N]
        Value Cinit = op.getOutputs()[0];  // [batch x M x N]

        // Require ranked tensor operands (bufferization must not have run yet)
        if (!isa<RankedTensorType>(A.getType()) || !isa<RankedTensorType>(B.getType()) ||
            !isa<RankedTensorType>(Cinit.getType()))
            return rewriter.notifyMatchFailure(
                op, "expected ranked tensor operands; already bufferized?");

        auto tensorA = cast<RankedTensorType>(A.getType());
        if (!tensorA.getElementType().isF32())
            return rewriter.notifyMatchFailure(op, "only f32 supported");

        auto cType = cast<RankedTensorType>(Cinit.getType());

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

        // Dimension sizes from A and B
        //   A[batch x M x K]  ->  dim 0 = batch, dim 1 = M, dim 2 = K
        //   B[batch x K x N]  ->  dim 2 = N
        Value batch = tensor::DimOp::create(rewriter, loc, A, 0);
        Value M = tensor::DimOp::create(rewriter, loc, A, 1);
        Value K = tensor::DimOp::create(rewriter, loc, A, 2);
        Value N = tensor::DimOp::create(rewriter, loc, B, 2);

        // -- b loop (outermost) ---------------------------------------------------
        // iter_arg: accumulated output tensor threaded through all loop levels
        auto bLoop = scf::ForOp::create(
            rewriter, loc, c0, batch, c1, /*iterArgs=*/ValueRange{ Cinit },
            [&](OpBuilder& b, Location loc, Value bv, ValueRange bArgs) {
                Value Cb = bArgs[0];

                // -- i loop ---------------------------------------------------------
                auto iLoop = scf::ForOp::create(
                    b, loc, c0, M, c1, ValueRange{ Cb },
                    [&](OpBuilder& b, Location loc, Value i, ValueRange iArgs) {
                        Value Ci = iArgs[0];

                        // -- k loop ---------------------------------------------------
                        auto kLoop = scf::ForOp::create(
                            b, loc, c0, K, c1, ValueRange{ Ci },
                            [&](OpBuilder& b, Location loc, Value k, ValueRange kArgs) {
                                Value Ck = kArgs[0];

                                // A[b, i, k] is invariant w.r.t. j — extract once and
                                // let the register allocator keep it live across j loop.
                                Value a =
                                    tensor::ExtractOp::create(b, loc, A, ValueRange{ bv, i, k });

                                // -- j loop (innermost) ----------------------------------
                                auto jLoop = scf::ForOp::create(
                                    b, loc, c0, N, c1, ValueRange{ Ck },
                                    [&](OpBuilder& b, Location loc, Value j, ValueRange jArgs) {
                                        Value Cj = jArgs[0];

                                        // B[b,k,j]: j increments along a row -> sequential
                                        Value bval = tensor::ExtractOp::create(
                                            b, loc, B, ValueRange{ bv, k, j });

                                        // C[b,i,j]: j increments along a row -> sequential
                                        Value cval = tensor::ExtractOp::create(
                                            b, loc, Cj, ValueRange{ bv, i, j });

                                        Value mul = arith::MulFOp::create(b, loc, a, bval);
                                        Value acc = arith::AddFOp::create(b, loc, cval, mul);

                                        // Produce updated tensor SSA value
                                        Value Cnew = tensor::InsertOp::create(
                                            b, loc, acc, Cj, ValueRange{ bv, i, j });

                                        scf::YieldOp::create(b, loc, Cnew);
                                    });

                                scf::YieldOp::create(b, loc, jLoop.getResult(0));
                            });

                        scf::YieldOp::create(b, loc, kLoop.getResult(0));
                    });

                scf::YieldOp::create(b, loc, iLoop.getResult(0));
            });

        rewriter.replaceOp(op, bLoop.getResult(0));
        return success();
    }
};

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

struct LinalgRewrite : public impl::LinalgRewriteBase<LinalgRewrite> {
    using LinalgRewriteBase::LinalgRewriteBase;

    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = funcOp.getContext();

        RewritePatternSet patterns(context);
        patterns.add<LinalgMatmulPattern, LinalgBatchMatmulPattern>(context);

        GreedyRewriteConfig config;
        config.setMaxIterations(1);
        config.setUseTopDownTraversal(true);

        (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
    }
};

}  // namespace mlir::bpa
