#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "Dialect/DIA/DIAOps.h"
#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {
struct DIATransposePattern : public OpRewritePattern<dia::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(dia::TransposeOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        if (opBandInfo.isDiagonal()) {
            rewriter.replaceOp(op, op.getInput());
            return success();
        } else
            return diaBandedTranspose(op, rewriter);
    }

    LogicalResult diaBandedTranspose(dia::TransposeOp op, PatternRewriter& rewriter) const {
        auto input = op.getInput();
        Operation* defInput = input.getDefiningOp();

        auto dict = defInput->getAttrDictionary();
        if (!dict) return failure();

        const BandedSubMatrix inputBand = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        const uint64_t lower = inputBand.Property.LowerBandwidth;
        const uint64_t upper = inputBand.Property.UpperBandwidth;

        auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
        const int64_t N = resultType.getDimSize(0);
        const int64_t M = resultType.getDimSize(1);

        Location loc = op->getLoc();

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimM = arith::ConstantIndexOp::create(rewriter, loc, M);

        Value lowerBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)lower);
        Value upperBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)upper);

        Value K =
            arith::ConstantIndexOp::create(rewriter, loc, static_cast<int64_t>(lower + upper + 1));
        Value uPlusL =
            arith::ConstantIndexOp::create(rewriter, loc, static_cast<int64_t>(lower + upper));

        // result
        Value emptyTensor = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});
        auto elementType = resultType.getElementType();
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ emptyTensor })
                .getResult(0);

        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, K, c1, ValueRange{ zeroedC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                Value cOut = iArgs[0];
                // jStart = max(0, l - i)
                Value lMinusI = arith::SubIOp::create(ob, loc, lowerBW, i);
                Value jStart = arith::MaxSIOp::create(ob, loc, lMinusI, c0);
                // end = M - max(0, i - l)
                Value iMinusL = arith::SubIOp::create(ob, loc, i, lowerBW);
                Value iMinusLClamped = arith::MaxSIOp::create(ob, loc, iMinusL, c0);
                Value jEnd = arith::SubIOp::create(ob, loc, dimM, iMinusLClamped);
                // ni = u + l - i
                Value newI = arith::SubIOp::create(ob, loc, uPlusL, i);
                auto jLoop = scf::ForOp::create(
                    ob, loc, jStart, jEnd, c1, ValueRange{ cOut },
                    [&](OpBuilder& ib, Location loc, Value j, ValueRange jArgs) {
                        Value cIn = jArgs[0];
                        // nj = j + iMinusL
                        Value newJ = arith::AddIOp::create(ib, loc, iMinusL, j);
                        Value val = tensor::ExtractOp::create(ib, loc, input, ValueRange{ i, j });
                        Value updated =
                            tensor::InsertOp::create(ib, loc, val, cIn, ValueRange{ newI, newJ });
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, jLoop.getResults());
            });
        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }
};

void addDIATransposePatterns(RewritePatternSet& patterns) {
    patterns.add<DIATransposePattern>(patterns.getContext());
}

}  // namespace mlir::bpa
