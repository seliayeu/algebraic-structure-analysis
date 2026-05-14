#include "Transform/BandedPropagation.h"

#include <cstdint>
#include <vector>

#include "Analysis/BandedStructureAnalysis.h"
#include "Dialect/DIA/DIAOps.h"
#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::bpa {

#define GEN_PASS_DEF_BANDEDANALYSIS
#include "lib/Transform/Passes.h.inc"

struct BandedAnalysisPass : public impl::BandedAnalysisBase<BandedAnalysisPass> {
    using BandedAnalysisBase::BandedAnalysisBase;

    LogicalResult updateShape(dia::MatmulOp op, BandedStructureAnalysis& BSA) {
        int64_t N = cast<RankedTensorType>(op.getOperand(1).getType()).getDimSize(1);

        auto result{ op.getResult() };
        auto output{ op.getOutput() };

        if (cast<RankedTensorType>(result.getType()).hasStaticShape()) return failure();
        if (!BSA.hasProperty(result)) return failure();

        BandedSubMatrix analysisResult{ BSA.getProperty(result) };

        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType) return failure();

        int64_t M = 0;
        if (!analysisResult.IsDia)
            M = N;
        else {
            int64_t lC = static_cast<int64_t>(analysisResult.Property.UpperBandwidth);
            int64_t uC = static_cast<int64_t>(analysisResult.Property.LowerBandwidth);
            M = std::min(2 * N - 1, lC + uC + 1);
        }

        auto newType = RankedTensorType::get({ M, N }, resultType.getElementType());
        result.setType(newType);
        output.setType(newType);
        if (auto emptyOp = output.getDefiningOp<tensor::EmptyOp>()) {
            emptyOp.getResult().setType(newType);
            emptyOp->setOperands({});
        }
        return success();
    }

    LogicalResult updateShape(dia::BatchMatmulOp op, BandedStructureAnalysis& BSA) {
        auto rank{ cast<RankedTensorType>(op.getOperand(1).getType()).getRank() };
        int64_t N = cast<RankedTensorType>(op.getOperand(1).getType()).getDimSize(rank - 1);

        auto result{ op.getResult() };
        auto output{ op.getOutput() };
        if (cast<RankedTensorType>(result.getType()).hasStaticShape()) return failure();
        if (!BSA.hasProperty(result)) return failure();

        BandedSubMatrix analysisResult{ BSA.getProperty(result) };

        auto resultType{ dyn_cast<RankedTensorType>(result.getType()) };
        if (!resultType) return failure();

        int64_t M = 0;
        if (!analysisResult.IsDia)
            M = N;
        else {
            const int64_t lC = static_cast<int64_t>(analysisResult.Property.UpperBandwidth);
            const int64_t uC = static_cast<int64_t>(analysisResult.Property.LowerBandwidth);
            M = std::min(2 * N - 1, lC + uC + 1);
        }

        std::vector<int64_t> dims;
        dims.reserve(resultType.getRank());
        if (resultType.getRank() > 2) {
            for (int64_t i = 0; i < resultType.getRank() - 2; ++i)
                dims.push_back(resultType.getDimSize(i));
        }
        dims.push_back(M);
        dims.push_back(N);

        auto newType = RankedTensorType::get(dims, resultType.getElementType());
        result.setType(newType);
        output.setType(newType);
        if (auto emptyOp = output.getDefiningOp<tensor::EmptyOp>()) {
            emptyOp.getResult().setType(newType);
            emptyOp->setOperands({});
        }
        return success();
    }

    LogicalResult updateShape(dia::ElementwiseOp op, BandedStructureAnalysis& BSA) {
        auto rank{ cast<RankedTensorType>(op.getOperand(0).getType()).getRank() };
        int64_t N = cast<RankedTensorType>(op.getOperand(0).getType()).getDimSize(rank - 1);

        auto result{ op.getResult() };
        auto output{ op.getOutput() };
        if (cast<RankedTensorType>(result.getType()).hasStaticShape()) return failure();
        if (!BSA.hasProperty(result)) return failure();

        BandedSubMatrix analysisResult{ BSA.getProperty(result) };

        auto resultType{ dyn_cast<RankedTensorType>(result.getType()) };
        if (!resultType) return failure();

        int64_t M = 0;
        if (!analysisResult.IsDia)
            M = N;
        else {
            int64_t lC = static_cast<int64_t>(analysisResult.Property.UpperBandwidth);
            int64_t uC = static_cast<int64_t>(analysisResult.Property.LowerBandwidth);
            M = std::min(2 * N - 1, lC + uC + 1);
        }

        std::vector<int64_t> dims;
        dims.reserve(resultType.getRank());
        if (resultType.getRank() > 2) {
            for (int64_t i = 0; i < resultType.getRank() - 2; ++i)
                dims.push_back(resultType.getDimSize(i));
        }
        dims.push_back(M);
        dims.push_back(N);

        auto newType = RankedTensorType::get(dims, resultType.getElementType());
        result.setType(newType);
        output.setType(newType);
        if (auto emptyOp = output.getDefiningOp<tensor::EmptyOp>()) {
            emptyOp.getResult().setType(newType);
            emptyOp->setOperands({});
        }
        return success();
    }

    LogicalResult updateShape(dia::SoftmaxOp op, BandedStructureAnalysis& BSA) {
        auto rank{ cast<RankedTensorType>(op.getInput().getType()).getRank() };
        int64_t N = cast<RankedTensorType>(op.getInput().getType()).getDimSize(rank - 1);

        auto result{ op.getResult() };
        auto output{ op.getOutput() };
        if (cast<RankedTensorType>(result.getType()).hasStaticShape()) return failure();
        if (!BSA.hasProperty(result)) return failure();

        BandedSubMatrix analysisResult{ BSA.getProperty(result) };

        auto resultType{ dyn_cast<RankedTensorType>(result.getType()) };
        if (!resultType) return failure();

        int64_t M = 0;
        if (!analysisResult.IsDia)
            M = N;
        else {
            int64_t lC = static_cast<int64_t>(analysisResult.Property.UpperBandwidth);
            int64_t uC = static_cast<int64_t>(analysisResult.Property.LowerBandwidth);
            M = std::min(2 * N - 1, lC + uC + 1);
        }

        std::vector<int64_t> dims;
        dims.reserve(resultType.getRank());
        if (resultType.getRank() > 2) {
            for (int64_t i = 0; i < resultType.getRank() - 2; ++i)
                dims.push_back(resultType.getDimSize(i));
        }
        dims.push_back(M);
        dims.push_back(N);

        auto newType = RankedTensorType::get(dims, resultType.getElementType());
        result.setType(newType);
        output.setType(newType);
        if (auto emptyOp = output.getDefiningOp<tensor::EmptyOp>()) {
            emptyOp.getResult().setType(newType);
            emptyOp->setOperands({});
        }
        return success();
    }

    LogicalResult updateShape(dia::TransposeOp op, BandedStructureAnalysis& BSA) {
        auto rank{ cast<RankedTensorType>(op.getInput().getType()).getRank() };
        int64_t N = cast<RankedTensorType>(op.getInput().getType()).getDimSize(rank - 1);

        auto result{ op.getResult() };
        auto output{ op.getOutput() };
        if (cast<RankedTensorType>(result.getType()).hasStaticShape()) return failure();
        if (!BSA.hasProperty(result)) return failure();

        BandedSubMatrix analysisResult{ BSA.getProperty(result) };

        auto resultType{ dyn_cast<RankedTensorType>(result.getType()) };
        if (!resultType) return failure();

        int64_t M = 0;
        if (!analysisResult.IsDia)
            M = N;
        else {
            int64_t lC = static_cast<int64_t>(analysisResult.Property.UpperBandwidth);
            int64_t uC = static_cast<int64_t>(analysisResult.Property.LowerBandwidth);
            M = std::min(2 * N - 1, lC + uC + 1);
        }

        std::vector<int64_t> dims;
        dims.reserve(resultType.getRank());
        if (resultType.getRank() > 2) {
            for (int64_t i = 0; i < resultType.getRank() - 2; ++i)
                dims.push_back(resultType.getDimSize(i));
        }
        dims.push_back(M);
        dims.push_back(N);

        auto newType = RankedTensorType::get(dims, resultType.getElementType());
        result.setType(newType);
        output.setType(newType);
        if (auto emptyOp = output.getDefiningOp<tensor::EmptyOp>()) {
            emptyOp.getResult().setType(newType);
            emptyOp->setOperands({});
        }
        return success();
    }

    LogicalResult updateShape(arith::ConstantOp op, const BandedStructureAnalysis& BSA) {
        auto oldType{ cast<RankedTensorType>(op.getResult().getType()) };
        int64_t N = oldType.getShape().back();

        auto result{ op.getResult() };
        if (!BSA.hasProperty(result)) return failure();
        BandedSubMatrix analysisResult{ BSA.getProperty(result) };
        BandedSubMatrix originalMat{ BSA.getOriginalProperty(result) };
        if (!analysisResult.IsDia) return failure();

        auto oldAttr{ cast<ElementsAttr>(op.getValue()) };
        auto oldValuesRange{ oldAttr.getValues<float>() };
        llvm::SmallVector<float> oldValues(oldValuesRange.begin(), oldValuesRange.end());

        auto newType{ cast<RankedTensorType>(op.getResult().getType()) };
        llvm::SmallVector<float> newValues;

        if (oldAttr.isSplat()) {
            auto newAttr = DenseElementsAttr::get(newType, oldAttr.getSplatValue<float>());
            op->setAttr("value", newAttr);
            return failure();
        }

        newValues.resize(newType.getNumElements(), 0.0f);

        auto oldL{ std::min(N - 1, static_cast<int64_t>(originalMat.Property.LowerBandwidth)) };
        auto oldU{ std::min(N - 1, static_cast<int64_t>(originalMat.Property.UpperBandwidth)) };
        auto newL{ std::min(N - 1, static_cast<int64_t>(analysisResult.Property.LowerBandwidth)) };
        auto newU{ std::min(N - 1, static_cast<int64_t>(analysisResult.Property.UpperBandwidth)) };

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
        op->setAttr("value", newAttr);
        return success();
    }

    void runOnOperation() override {
#ifdef ENABLE_BENCHMARKING
        auto start{ std::chrono::high_resolution_clock::now() };
#endif
        auto funcOp{ getOperation() };
        auto* context{ funcOp->getContext() };
        BandedStructureAnalysis BSA(detectDIA);

        for (auto& block : funcOp.getBody()) (void)BSA.run(&block);

        Builder builder(context);

        funcOp->walk([&](Operation* inst) {
            auto results{ inst->getResults() };
            if (results.size() != 1 || !BSA.hasProperty(results[0])) return;

            BandedSubMatrix& analysisResult{ BSA.getProperty(results[0]) };
            auto property{ analysisResult.Property };
            auto dims{ analysisResult.Dims };

            auto resultType = dyn_cast<RankedTensorType>(results[0].getType());
            if (!resultType) return;

            const intptr_t rank = resultType.getRank();
            const uint64_t N = rank == 3 ? resultType.getDimSize(2) : resultType.getDimSize(1);

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
            if (!dyn_cast<dia::SoftmaxOp>(inst) && !dyn_cast<dia::MatmulOp>(inst) &&
                !dyn_cast<dia::ElementwiseOp>(inst) && !dyn_cast<dia::BatchMatmulOp>(inst) &&
                !dyn_cast<dia::TransposeOp>(inst) &&
                !(dyn_cast<arith::ConstantOp>(inst) &&
                  isa<RankedTensorType>(inst->getResult(0).getType())))
                return;

            if (auto matmulOp{ dyn_cast<dia::MatmulOp>(inst) }) {
                if (failed(updateShape(matmulOp, BSA))) return;
            } else if (auto batchMatmulOp{ dyn_cast<dia::BatchMatmulOp>(inst) }) {
                if (failed(updateShape(batchMatmulOp, BSA))) return;
            } else if (auto elementwiseOp{ dyn_cast<dia::ElementwiseOp>(inst) }) {
                if (failed(updateShape(elementwiseOp, BSA))) return;
            } else if (auto transposeOp{ dyn_cast<dia::TransposeOp>(inst) }) {
                if (failed(updateShape(transposeOp, BSA))) return;
            } else if (auto constantOp{ dyn_cast<arith::ConstantOp>(inst) }) {
                if (failed(updateShape(constantOp, BSA))) return;
            } else if (auto softmaxOp{ dyn_cast<dia::SoftmaxOp>(inst) }) {
                if (failed(updateShape(softmaxOp, BSA))) return;

            } else {
                return;
            }
        });

        if (heuristic) {
            SmallVector<dia::MatmulOp> matmuls;
            funcOp->walk([&](Operation* inst) {
                if (!isa<dia::MatmulOp>(inst)) return;
                auto matmulOp{ cast<dia::MatmulOp>(inst) };
                auto lhs{ matmulOp.getLhs() };
                auto rhs{ matmulOp.getRhs() };
                auto result{ matmulOp.getResult() };

                BandedSubMatrix lhsRes{ BSA.getProperty(lhs) };
                BandedSubMatrix rhsRes{ BSA.getProperty(rhs) };
                BandedSubMatrix resultRes{ BSA.getProperty(result) };

                if (!resultRes.IsDia) return;

                if (!lhsRes.IsDia || !rhsRes.IsDia) matmuls.push_back(matmulOp);
            });

            for (auto& matmulOp : matmuls) {
                OpBuilder builder(matmulOp);
                auto lhs{ matmulOp.getLhs() };
                auto rhs{ matmulOp.getRhs() };
                BandedSubMatrix& lhsRes{ BSA.getProperty(lhs) };
                BandedSubMatrix& rhsRes{ BSA.getProperty(rhs) };
                if (!lhsRes.IsDia) {
                    auto lhsType{ cast<RankedTensorType>(lhs.getType()) };
                    auto rows{ std::min(lhsRes.Property.UpperBandwidth,
                                        static_cast<uint64_t>(lhsType.getDimSize(1) - 1)) +
                               std::min(lhsRes.Property.LowerBandwidth,
                                        static_cast<uint64_t>(lhsType.getDimSize(1) - 1)) +
                               1 };
                    auto diaType{ RankedTensorType::get(
                        { static_cast<int64_t>(rows), lhsType.getDimSize(1) },
                        lhsType.getElementType()) };
                    auto fromDenseOp{ dia::FromDenseOp::create(builder, matmulOp->getLoc(), diaType,
                                                               lhs) };
                    auto newMat{ lhsRes };
                    newMat.IsDia = true;
                    BSA.setProperty(fromDenseOp, newMat);
                    auto attrDict{ cast<DictionaryAttr>(lhs.getDefiningOp()->getAttr("metadata")) };
                    SmallVector<NamedAttribute> attrs(attrDict.begin(), attrDict.end());
                    attrs.push_back(builder.getNamedAttr("dia", builder.getBoolAttr(true)));
                    auto newDict{ DictionaryAttr::get(builder.getContext(), attrs) };
                    fromDenseOp->setAttr("metadata", newDict);

                    matmulOp.getLhsMutable().assign(fromDenseOp.getResult());
                }
                if (!rhsRes.IsDia) {
                    auto rhsType{ cast<RankedTensorType>(rhs.getType()) };
                    auto rows{ std::min(rhsRes.Property.UpperBandwidth,
                                        static_cast<uint64_t>(rhsType.getDimSize(1) - 1)) +
                               std::min(rhsRes.Property.LowerBandwidth,
                                        static_cast<uint64_t>(rhsType.getDimSize(1) - 1)) +
                               1 };
                    auto diaType{ RankedTensorType::get(
                        { static_cast<int64_t>(rows), rhsType.getDimSize(1) },
                        rhsType.getElementType()) };
                    auto fromDenseOp{ dia::FromDenseOp::create(builder, matmulOp->getLoc(), diaType,
                                                               rhs) };
                    auto newMat{ rhsRes };
                    newMat.IsDia = true;
                    BSA.setProperty(fromDenseOp, newMat);
                    auto attrDict{ cast<DictionaryAttr>(rhs.getDefiningOp()->getAttr("metadata")) };
                    SmallVector<NamedAttribute> attrs(attrDict.begin(), attrDict.end());
                    attrs.push_back(builder.getNamedAttr("dia", builder.getBoolAttr(true)));
                    auto newDict{ DictionaryAttr::get(builder.getContext(), attrs) };
                    fromDenseOp->setAttr("metadata", newDict);
                    matmulOp.getRhsMutable().assign(fromDenseOp.getResult());
                }
            }
        }

        auto& result = getAnalysis<BandedAnalysisResult>();
        result.detectDIA = detectDIA;
        markAnalysesPreserved<BandedAnalysisResult>();
#ifdef ENABLE_BENCHMARKING
        auto end{ std::chrono::high_resolution_clock::now() };
        llvm::errs() << "BandedAnalysis time: "
                     << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end -
                                                                                              start)
                            .count()
                     << " ms\n";
#endif
    }
};

}  // namespace mlir::bpa
