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
        UpperTriangular,
        LowerTriangular,
        General,
    };

    AlgebraicStructureAnalysisLatticeValue(AlgebraicProperty state) : state{ state } {}
    AlgebraicStructureAnalysisLatticeValue() : state{ AlgebraicProperty::Uninitialized } {}

    static AlgebraicStructureAnalysisLatticeValue join(const AlgebraicStructureAnalysisLatticeValue& lhs, const AlgebraicStructureAnalysisLatticeValue& rhs) {
        if (lhs.state == AlgebraicProperty::Uninitialized)
            return rhs;
        if (rhs.state == AlgebraicProperty::Uninitialized)
            return lhs;
        if (lhs.state == AlgebraicProperty::General)
            return rhs;
        if (rhs.state == AlgebraicProperty::General)
            return lhs;
        if (lhs.state == AlgebraicProperty::Symmetric || lhs.state == AlgebraicProperty::UpperTriangular || lhs.state == AlgebraicProperty::LowerTriangular)
            return AlgebraicProperty::Diagonal; // rhs is either sym, upper, lower, or diagonal. either way, output is diagonal
        if (rhs.state == AlgebraicProperty::Symmetric || rhs.state == AlgebraicProperty::UpperTriangular || rhs.state == AlgebraicProperty::LowerTriangular)
            return AlgebraicProperty::Diagonal; // rhs is either sym, upper, lower, or diagonal. either way, output is diagonal
        if (lhs.state == AlgebraicProperty::Diagonal)
            return rhs;
        if (rhs.state == AlgebraicProperty::Diagonal)
            return lhs;
        return lhs; // identity
    }

    static AlgebraicStructureAnalysisLatticeValue meet(const AlgebraicStructureAnalysisLatticeValue& lhs, const AlgebraicStructureAnalysisLatticeValue& rhs) {
        assert(lhs.state != AlgebraicProperty::Uninitialized && rhs.state != AlgebraicProperty::Uninitialized && "All states should be initialized before using meet");
        if (lhs.state == AlgebraicProperty::Identity)
            return rhs;
        if (rhs.state == AlgebraicProperty::Identity)
            return lhs;
        if (lhs.state == AlgebraicProperty::Diagonal)
            return rhs;
        if (rhs.state == AlgebraicProperty::Diagonal)
            return lhs;
        if (lhs.state == AlgebraicProperty::Symmetric || lhs.state == AlgebraicProperty::UpperTriangular || lhs.state == AlgebraicProperty::LowerTriangular)
            return AlgebraicProperty::General; // rhs is either sym, upper, lower, or diagonal. either way, output is diagonal
        if (rhs.state == AlgebraicProperty::Symmetric || rhs.state == AlgebraicProperty::UpperTriangular || rhs.state == AlgebraicProperty::LowerTriangular)
            return AlgebraicProperty::General; // rhs is either sym, upper, lower, or diagonal. either way, output is diagonal
        return lhs; // identity
    }

    bool operator==(const AlgebraicStructureAnalysisLatticeValue& rhs) const {
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
            case AlgebraicProperty::UpperTriangular:
                return "UpperTriangular";
            case AlgebraicProperty::LowerTriangular:
                return "LowerTriangular";
            case AlgebraicProperty::General:
                return "General";
        }
        return "General";
    }

    static const AlgebraicProperty stringRefAsValue(StringRef value) {
        static const DenseMap<StringRef, AlgebraicProperty> dict {
            { "Uninitialized", AlgebraicProperty::Uninitialized },
            { "Identity", AlgebraicProperty::Identity },
            { "Diagonal", AlgebraicProperty::Diagonal },
            { "Symmetric", AlgebraicProperty::Symmetric },
            { "UpperTriangular", AlgebraicProperty::UpperTriangular },
            { "LowerTriangular", AlgebraicProperty::LowerTriangular },
            { "General", AlgebraicProperty::General },
        };
        return dict.contains(value) ? dict.at(value) : AlgebraicProperty::General;
    }

    static AlgebraicProperty binaryMatmul(AlgebraicProperty lhs, AlgebraicProperty rhs) {
        if (lhs == AlgebraicProperty::Identity && lhs == rhs)
            return AlgebraicProperty::Identity;
        if (lhs == AlgebraicProperty::Diagonal && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::LowerTriangular && lhs == rhs)
            return AlgebraicProperty::LowerTriangular;
        if (lhs == AlgebraicProperty::UpperTriangular && lhs == rhs)
            return AlgebraicProperty::UpperTriangular;
        return AlgebraicProperty::General;
    }

    static AlgebraicProperty binaryAdd(AlgebraicProperty lhs, AlgebraicProperty rhs) {
        if (lhs == AlgebraicProperty::Identity && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::Diagonal && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::Symmetric && lhs == rhs)
            return AlgebraicProperty::Symmetric;
        if (lhs == AlgebraicProperty::LowerTriangular && lhs == rhs)
            return AlgebraicProperty::LowerTriangular;
        if (lhs == AlgebraicProperty::UpperTriangular && lhs == rhs)
            return AlgebraicProperty::UpperTriangular;
        return AlgebraicProperty::General;
    }

    static AlgebraicProperty binaryElementwiseGeneral(AlgebraicProperty lhs, AlgebraicProperty rhs) {
        if (lhs == AlgebraicProperty::Identity && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::Diagonal && lhs == rhs)
            return AlgebraicProperty::Diagonal;
        if (lhs == AlgebraicProperty::Symmetric && lhs == rhs)
            return AlgebraicProperty::Symmetric;
        return AlgebraicProperty::General;
    }

    static AlgebraicProperty binaryElementwiseProduct(AlgebraicProperty lhs, AlgebraicProperty rhs) {
        return meet(lhs, rhs).state;
    }

    static AlgebraicProperty unaryElementwise(AlgebraicProperty property) {
        if (property == AlgebraicProperty::Identity)
            return AlgebraicProperty::Diagonal;
        return property;
    }

    static AlgebraicProperty transpose(AlgebraicProperty property) {
        if (property == AlgebraicProperty::UpperTriangular)
            return AlgebraicProperty::LowerTriangular;
        if (property == AlgebraicProperty::LowerTriangular)
            return AlgebraicProperty::UpperTriangular;
        return property;
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

// class AlgebraicBackwardStructureAnalysis : public dataflow::SparseBackwardDataFlowAnalysis<dataflow::Lattice<AlgebraicStructureAnalysisLatticeValue>> {
// public:
//     using AlgebraicStructureAnalysisLattice = dataflow::Lattice<AlgebraicStructureAnalysisLatticeValue>;
//     using AlgebraicProperty = AlgebraicStructureAnalysisLatticeValue::AlgebraicProperty;
//
//     using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
//     LogicalResult visitOperation(Operation *op, ArrayRef<AlgebraicStructureAnalysisLattice*> operands, ArrayRef<const AlgebraicStructureAnalysisLattice*> results) override;
// private:
//     void setToEntryState(AlgebraicStructureAnalysisLattice *lattice) override;
// };

}

#endif
