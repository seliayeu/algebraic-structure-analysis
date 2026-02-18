#include "Transform/Banded/BandedLoweringPass.h"

#include "lib/Analysis/BandedStructureAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bpa {
struct DiagonalMatmulPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        // only handle diagonals for now
        if (!(opBandInfo.Property.LowerBandwidth == opBandInfo.Property.UpperBandwidth == 0))
            return success();

        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        int64_t n = resultType.getDimSize(0);

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dim = arith::ConstantIndexOp::create(rewriter, loc, n);

        auto loop = scf::ForOp::create(
            rewriter, loc, c0, dim, c1, ValueRange{ C },
            [&](OpBuilder& b, Location loc, Value i, ValueRange iterArgs) {
                Value cOut = iterArgs[0];

                // Extract A[i,i] and B[i,i]
                Value aVal = tensor::ExtractOp::create(b, loc, A, ValueRange{ i, i });
                Value bVal = tensor::ExtractOp::create(b, loc, B, ValueRange{ i, i });
                // Scalar multiply
                Value mul = arith::MulFOp::create(b, loc, aVal, bVal);
                // Updates the C's placeholder
                Value updated = tensor::InsertOp::create(b, loc, mul, cOut, ValueRange{ i, i });
                // Yields the result to the next iteration
                scf::YieldOp::create(b, loc, updated);
            });

        rewriter.replaceOp(op, loop.getResult(0));

        return success();
    }
};

}  // namespace mlir::bpa
