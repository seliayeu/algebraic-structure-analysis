#include "Transform/BandedPropagation.h"

#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::bpa {

#define GEN_PASS_DEF_BANDEDANALYSIS
#include "lib/Transform/Passes.h.inc"

struct BandedAnalysisPass : public impl::BandedAnalysisBase<BandedAnalysisPass> {
    using BandedAnalysisBase::BandedAnalysisBase;

    void runOnOperation() override {
        auto funcOp{ getOperation() };
        auto* context{ funcOp->getContext() };
        BandedStructureAnalysis BSA(detectDIA);

        for (auto& block : funcOp.getBody()) (void)BSA.run(&block);

        Builder builder(context);

        funcOp->walk([&](Operation* inst) {
            auto results{ inst->getResults() };
            if (results.size() != 1 || !BSA.hasProperty(results[0])) return;

            BandedSubMatrix analysisResult{ BSA.getProperty(results[0]) };
            auto property{ analysisResult.Property };
            auto dims{ analysisResult.Dims };

            auto resultType = dyn_cast<RankedTensorType>(results[0].getType());
            if (!resultType) return;

            const uint64_t N = resultType.getDimSize(1);

            auto upperAttr{ builder.getNamedAttr(
                "upperBw", builder.getI64IntegerAttr(property.UpperBandwidth)) };

            auto lowerAttr{ builder.getNamedAttr(
                "lowerBw", builder.getI64IntegerAttr(property.LowerBandwidth)) };

            auto dim0Attr{ builder.getI64IntegerAttr(dims[0]) };
            auto dim1Attr{ builder.getI64IntegerAttr(dims[1]) };
            auto dimsArrayAttr{ builder.getNamedAttr(
                "propertyDims", builder.getArrayAttr({ dim0Attr, dim1Attr })) };

            llvm::SmallVector<mlir::NamedAttribute> attrs{ upperAttr, lowerAttr, dimsArrayAttr };

            if (detectDIA) {
                if (shouldCompressResult(*inst, analysisResult, N)) {
                    attrs.emplace_back(builder.getNamedAttr("dia", builder.getBoolAttr(true)));
                    analysisResult.IsDia = true;
                }
            } else if (analysisResult.IsDia)
                attrs.emplace_back(builder.getNamedAttr("dia", builder.getBoolAttr(true)));

            auto dictAttr = builder.getDictionaryAttr(attrs);
            inst->setAttr("metadata", dictAttr);
        });
        // Propagates dia shapes
        // I believe that this second walk could be blended in the first by moving this logic
        // inside the visitOp functions
        funcOp->walk([&](Operation* inst) {
            auto matmulOp = dyn_cast<dia::MatmulOp>(inst);
            if (!matmulOp) return;

            auto results = inst->getResults();
            if (results.size() != 1) return;
            if (!BSA.hasProperty(results[0])) return;

            const BandedSubMatrix opBand = BSA.getProperty(results[0]);

            if (!opBand.IsDia) return;

            auto resultType = dyn_cast<RankedTensorType>(results[0].getType());
            if (!resultType) return;

            int64_t lC = static_cast<int64_t>(opBand.Property.UpperBandwidth);
            int64_t uC = static_cast<int64_t>(opBand.Property.LowerBandwidth);
            int64_t N = cast<RankedTensorType>(inst->getOperand(1).getType()).getDimSize(1);
            int64_t numDiags = std::min(2 * N - 1, lC + uC + 1);

            auto newType = RankedTensorType::get({ numDiags, N }, resultType.getElementType());
            results[0].setType(newType);

            Value outsVal = matmulOp.getOutput();
            outsVal.setType(newType);

            if (auto emptyOp = outsVal.getDefiningOp<tensor::EmptyOp>()) {
                emptyOp.getResult().setType(newType);
                emptyOp->setOperands({});
            }
        });
        // cache information
        auto& result = getAnalysis<BandedAnalysisResult>();
        result.detectDIA = detectDIA;
        markAnalysesPreserved<BandedAnalysisResult>();
    }
};

}  // namespace mlir::bpa
