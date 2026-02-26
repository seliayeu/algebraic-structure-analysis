#include "Analysis/BandedStructureAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::bpa {

inline bool shouldCompress(const BandedSubMatrix& resultBand, const uint64_t N) {
    return (resultBand.Property.LowerBandwidth + resultBand.Property.UpperBandwidth) < N;
}

DictionaryAttr getMetadataWithTensorLayout(Operation& op, PatternRewriter& rewriter,
                                           const StringRef& layout);
}  // namespace mlir::bpa
