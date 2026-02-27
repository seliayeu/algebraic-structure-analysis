#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

DictionaryAttr getMetadataWithTensorLayout(Operation& op, PatternRewriter& rewriter,
                                           const StringRef& layout) {
    DictionaryAttr oldMetadata = op.getAttrOfType<DictionaryAttr>("metadata");

    SmallVector<NamedAttribute> entries;

    if (oldMetadata) entries.append(oldMetadata.begin(), oldMetadata.end());

    entries.emplace_back(rewriter.getStringAttr("layout"), rewriter.getStringAttr(layout));

    DictionaryAttr newMetadata = rewriter.getDictionaryAttr(entries);
    return newMetadata;
}
}  // namespace mlir::bpa
