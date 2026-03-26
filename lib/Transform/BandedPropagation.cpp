#include "Transform/BandedPropagation.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include "Analysis/BandedStructureAnalysis.h"
#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::bpa {

#define GEN_PASS_DEF_BANDEDANALYSIS
#include "lib/Transform/Passes.h.inc"

struct BandedAnalysisPass : public impl::BandedAnalysisBase<BandedAnalysisPass> {
    using BandedAnalysisBase::BandedAnalysisBase;

    LogicalResult updateShape(TypedValue<RankedTensorType> result, int64_t N,
                              BandedStructureAnalysis& BSA) {
        if (!BSA.hasProperty(result)) return failure();

        BandedSubMatrix analysisResult{ BSA.getProperty(result) };

        if (!analysisResult.IsDia) return failure();

        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType) return failure();

        int64_t lC = static_cast<int64_t>(analysisResult.Property.UpperBandwidth);
        int64_t uC = static_cast<int64_t>(analysisResult.Property.LowerBandwidth);
        int64_t numDiags = std::min(2 * N - 1, lC + uC + 1);

        std::vector<int64_t> dims;
        dims.reserve(resultType.getRank());
        if (resultType.getRank() > 2) {
            for (int64_t i = 0; i < resultType.getRank() - 2; ++i)
                dims.push_back(resultType.getDimSize(i));
        }
        dims.push_back(numDiags);
        dims.push_back(N);

        auto newType = RankedTensorType::get(dims, resultType.getElementType());
        result.setType(newType);
        return success();
    }

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

        funcOp->walk([&](Operation* inst) {
            if (!dyn_cast<dia::MatmulOp>(inst) && !dyn_cast<dia::ElementwiseOp>(inst) &&
                !dyn_cast<dia::BatchMatmulOp>(inst) && !dyn_cast<dia::TransposeOp>(inst) &&
                !(dyn_cast<arith::ConstantOp>(inst) &&
                  isa<RankedTensorType>(inst->getResult(0).getType())))
                return;

            auto result{ inst->getResult(0) };

            Value output;
            if (auto matmulOp{ dyn_cast<dia::MatmulOp>(inst) }) {
                int64_t N = cast<RankedTensorType>(inst->getOperand(1).getType()).getDimSize(1);
                output = matmulOp.getOutput();
                if (failed(updateShape(matmulOp.getResult(), N, BSA))) return;
            } else if (auto batchMatmulOp{ dyn_cast<dia::BatchMatmulOp>(inst) }) {
                int64_t N = cast<RankedTensorType>(inst->getOperand(1).getType()).getDimSize(2);
                output = batchMatmulOp.getOutput();
                if (failed(updateShape(batchMatmulOp.getResult(), N, BSA))) return;
            } else if (auto elementwiseOp{ dyn_cast<dia::ElementwiseOp>(inst) }) {
                int64_t N = cast<RankedTensorType>(inst->getOperand(0).getType()).getDimSize(1);
                output = elementwiseOp.getOutput();
                if (failed(updateShape(elementwiseOp.getResult(), N, BSA))) return;
            } else if (auto transposeOp{ dyn_cast<dia::TransposeOp>(inst) }) {
                int64_t N = cast<RankedTensorType>(inst->getOperand(0).getType()).getDimSize(1);
                output = transposeOp.getOutput();
                if (failed(updateShape(transposeOp.getResult(), N, BSA))) return;
            } else if (auto constantOp{ dyn_cast<arith::ConstantOp>(inst) }) {
                auto oldType{ cast<RankedTensorType>(constantOp.getResult().getType()) };
                int64_t N = oldType.getShape().back();
                if (failed(updateShape(cast<TypedValue<RankedTensorType>>(constantOp.getResult()),
                                       N, BSA))) {
                    return;
                }

                auto oldAttr{ cast<ElementsAttr>(constantOp.getValue()) };
                auto oldValuesRange{ oldAttr.getValues<float>() };
                llvm::SmallVector<float> oldValues(oldValuesRange.begin(), oldValuesRange.end());

                auto newType{ cast<RankedTensorType>(constantOp.getResult().getType()) };
                llvm::SmallVector<float> newValues;

                if (oldAttr.isSplat()) {
                    auto newAttr = DenseElementsAttr::get(newType, oldAttr.getSplatValue<float>());
                    constantOp->setAttr("value", newAttr);
                    return;
                }

                newValues.resize(newType.getNumElements(), 0.0f);

                BandedSubMatrix analysisResult{ BSA.getProperty(result) };
                BandedSubMatrix originalMat{ BSA.getOriginalProperty(result) };
                auto oldL{ std::min(N - 1,
                                    static_cast<int64_t>(originalMat.Property.LowerBandwidth)) };
                auto oldU{ std::min(N - 1,
                                    static_cast<int64_t>(originalMat.Property.UpperBandwidth)) };
                auto newL{ std::min(N - 1,
                                    static_cast<int64_t>(analysisResult.Property.LowerBandwidth)) };
                auto newU{ std::min(N - 1,
                                    static_cast<int64_t>(analysisResult.Property.UpperBandwidth)) };

                auto oldDiags{ oldL + oldU + 1 };
                auto newDiags{ newL + newU + 1 };

                int64_t outerElements = oldType.getNumElements() / ((oldL + oldU + 1) * N);
                for (int64_t b = 0; b < outerElements; ++b) {
                    int64_t oldBatchOffset = b * (oldL + oldU + 1) * N;
                    int64_t newBatchOffset = b * (newL + newU + 1) * N;

                    for (int64_t newRow = 0; newRow < newDiags; ++newRow) {
                        int64_t d = newRow - newL;
                        int64_t oldRow = d + oldL;
                        if (oldRow >= 0 && oldRow < oldDiags) {
                            for (int64_t i = 0; i < N; ++i) {
                                newValues[newBatchOffset + newRow * N + i] =
                                    oldValues[oldBatchOffset + oldRow * N + i];
                            }
                        }
                    }
                }

                auto newAttr{ DenseElementsAttr::get(newType, llvm::ArrayRef<float>(newValues)) };
                constantOp->setAttr("value", newAttr);
                return;
            } else {
                return;
            }

            output.setType(cast<RankedTensorType>(result.getType()));
            if (auto emptyOp = output.getDefiningOp<tensor::EmptyOp>()) {
                emptyOp.getResult().setType(cast<RankedTensorType>(result.getType()));
                emptyOp->setOperands({});
            }
        });

        auto& result = getAnalysis<BandedAnalysisResult>();
        result.detectDIA = detectDIA;
        markAnalysesPreserved<BandedAnalysisResult>();
    }
};

}  // namespace mlir::bpa
