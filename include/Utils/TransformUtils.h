#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::bpa {

// Write result to DIA format only when:
// The result is already set to DIA
// Instruction is supported by the analysis (which means that only forward propagation change
// layouts
// Compress overhead is smaller than dense space waste
inline bool shouldCompressResult(Operation& op, const BandedSubMatrix& resultBand,
                                 const uint64_t N) {
    const bool isAlreadyDIA = resultBand.IsDia;
    auto dialect = op.getDialect();
    const bool isDialectSupported = (dialect && (dialect->getNamespace() == "dia"));
    const bool isOverheadSmaller =
        (resultBand.Property.LowerBandwidth + resultBand.Property.UpperBandwidth) < N - 1;
    return isAlreadyDIA || (isDialectSupported && isOverheadSmaller);
}

inline bool isFullyDense(const BandedSubMatrix& A, const BandedSubMatrix& B,
                         const BandedSubMatrix& C, const uint64_t maxBandwidth) {
    return A.Property.LowerBandwidth >= maxBandwidth && A.Property.UpperBandwidth >= maxBandwidth &&
           B.Property.LowerBandwidth >= maxBandwidth && B.Property.UpperBandwidth >= maxBandwidth &&
           C.Property.LowerBandwidth >= maxBandwidth && C.Property.UpperBandwidth >= maxBandwidth;
}

}  // namespace mlir::bpa
