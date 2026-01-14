#ifndef LIB_ANALYSIS_ALGEBRAICSTRUCTUREANALYSIS_H
#define LIB_ANALYSIS_ALGEBRAICSTRUCTUREANALYSIS_H
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
// #include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir::asa {

class AlgebraicStructureAnalysisLatticeValue {
public:
    enum class AlgebraicProperty : unsigned int {
        Uninitialized,
        Identity,
        Diagonal,
        Symmetric,
        Unknown,
    };

    AlgebraicStructureAnalysisLatticeValue(AlgebraicProperty state) : state{ state } {}
    AlgebraicStructureAnalysisLatticeValue() : state{ AlgebraicProperty::Uninitialized } {}

    static AlgebraicStructureAnalysisLatticeValue join(const AlgebraicStructureAnalysisLatticeValue& lhs, const AlgebraicStructureAnalysisLatticeValue& rhs) {
        if (lhs.state == AlgebraicProperty::Uninitialized)
            return rhs;
        if (rhs.state == AlgebraicProperty::Uninitialized)
            return lhs;
        if (lhs.state == AlgebraicProperty::Identity)
            return rhs;
        if (rhs.state == AlgebraicProperty::Identity)
            return lhs;
        if (lhs.state == AlgebraicProperty::Diagonal)
            return rhs;
        if (rhs.state == AlgebraicProperty::Diagonal)
            return lhs;
        if (lhs.state == AlgebraicProperty::Symmetric)
            return rhs;
        if (rhs.state == AlgebraicProperty::Symmetric)
            return lhs;
        return lhs; // Unknown
    }

    bool operator==(const AlgebraicStructureAnalysisLatticeValue& rhs) {
        return state == rhs.state;
    }

    static StringRef propertyAsStringRef(AlgebraicProperty property) {
        switch (property) {
            case AlgebraicProperty::Uninitialized:
                return "Uninitialized";
            case AlgebraicProperty::Identity:
                return "Identity";
            case AlgebraicProperty::Diagonal:
                return "Diagonal";
            case AlgebraicProperty::Symmetric:
                return "Symmetric";
            case AlgebraicProperty::Unknown:
                return "Unknown";
        }
        return "Unknown";
    }

    static const AlgebraicProperty stringRefAsValue(StringRef value) {
        static const DenseMap<StringRef, AlgebraicProperty> dict {
            { "Uninitialized", AlgebraicProperty::Uninitialized },
            { "Identity", AlgebraicProperty::Identity },
            { "Diagonal", AlgebraicProperty::Diagonal },
            { "Symmetric", AlgebraicProperty::Symmetric },
            { "Unknown", AlgebraicProperty::Unknown },
        };
        return dict.contains(value) ? dict.at(value) : AlgebraicProperty::Unknown;
    }

    static AlgebraicProperty binaryMatmul(AlgebraicProperty lhs, AlgebraicProperty rhs) {
        if (lhs == AlgebraicProperty::Identity && lhs == rhs)
            return AlgebraicProperty::Identity;
        if (lhs == AlgebraicProperty::Diagonal && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        return AlgebraicProperty::Unknown;
    }

    static AlgebraicProperty binaryAdd(AlgebraicProperty lhs, AlgebraicProperty rhs) {
        if (lhs == AlgebraicProperty::Identity && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::Diagonal && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::Symmetric && lhs == rhs)
            return AlgebraicProperty::Symmetric;
        return AlgebraicProperty::Unknown;
    }

    static AlgebraicProperty binaryElementwise(AlgebraicProperty lhs, AlgebraicProperty rhs) {
        if (lhs == AlgebraicProperty::Identity && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::Diagonal && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::Symmetric && lhs == rhs)
            return AlgebraicProperty::Symmetric;
        return AlgebraicProperty::Unknown;
    }

    AlgebraicProperty getState() const { return state; };

    void print(raw_ostream& os) const { os << propertyAsStringRef(state); }

private:
    AlgebraicProperty state{ AlgebraicProperty::Uninitialized };
};

class AlgebraicStructureAnalysis : public dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<AlgebraicStructureAnalysisLatticeValue>> {
public:
    using AlgebraicStructureAnalysisLattice = dataflow::Lattice<AlgebraicStructureAnalysisLatticeValue>;
    using AlgebraicProperty = AlgebraicStructureAnalysisLatticeValue::AlgebraicProperty;

    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
    LogicalResult visitOperation(Operation *op, ArrayRef<const AlgebraicStructureAnalysisLattice*> operands, ArrayRef<AlgebraicStructureAnalysisLattice*> results) override;
private:
    void setToEntryState(AlgebraicStructureAnalysisLattice *lattice) override;
};

}
#endif
