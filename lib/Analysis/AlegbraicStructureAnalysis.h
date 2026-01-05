#ifndef LIB_ANALYSIS_ALGEBRAICSTRUCTUREANALYSIS_H
#define LIB_ANALYSIS_ALGEBRAICSTRUCTUREANALYSIS_H
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir::asa {

class AlgebraicStructureAnalysisState : public AnalysisState {
public:
    using AnalysisState::AnalysisState;

    enum class ASAValue : unsigned int {
        Identity,
        Diagonal,
        Symmetric,
        Unknown,
    };

    static StringRef valueAsString(ASAValue value) {
        switch (value) {
            case ASAValue::Identity:
                return "Identity";
            case ASAValue::Diagonal:
                return "Diagonal";
            case ASAValue::Symmetric:
                return "Symmetric";
            case ASAValue::Unknown:
                return "Unknown";
        }
        return "Unknown";
    }

    static const ASAValue stringAsValue(StringRef value) {
        static const DenseMap<StringRef, ASAValue> dict {
            { "Identity", ASAValue::Identity },
            { "Diagonal", ASAValue::Diagonal },
            { "Symmetric", ASAValue::Symmetric },
            { "Unknown", ASAValue::Unknown },
        };
        return dict.contains(value) ? dict.at(value) : ASAValue::Unknown;
    }

    static ASAValue binMatmul(ASAValue lhs, ASAValue rhs) {
        if (lhs == ASAValue::Identity && lhs == rhs)
            return ASAValue::Identity;
        if (lhs == ASAValue::Diagonal && lhs == rhs)
            return ASAValue::Diagonal;
        return ASAValue::Unknown;
    }

    static ASAValue binAdd(ASAValue lhs, ASAValue rhs) {
        if (lhs == ASAValue::Identity && lhs == rhs)
            return ASAValue::Diagonal;
        if (lhs == ASAValue::Diagonal && lhs == rhs)
            return ASAValue::Diagonal;
        if (lhs == ASAValue::Symmetric && lhs == rhs)
            return ASAValue::Symmetric;
        return ASAValue::Unknown;
    }

    static ASAValue binElementwise(ASAValue lhs, ASAValue rhs) {
        if (lhs == ASAValue::Identity && lhs == rhs)
            return ASAValue::Diagonal;
        if (lhs == ASAValue::Diagonal && lhs == rhs)
            return ASAValue::Diagonal;
        if (lhs == ASAValue::Symmetric && lhs == rhs)
            return ASAValue::Symmetric;
        return ASAValue::Unknown;
    }

    void print(raw_ostream &os) const override { os << "(" << valueAsString(value) << ")"; };
    ASAValue getValue() const { return value; }
    ChangeResult setValue(ASAValue newValue) { 
        auto result{ value == newValue ? ChangeResult::NoChange : ChangeResult::Change };
        value = newValue; 
        return result;
    }
private:
    ASAValue value{ ASAValue::Unknown };
};

class AlgebraicStructureAnalysis : public DataFlowAnalysis {
public:
    explicit AlgebraicStructureAnalysis(DataFlowSolver &solver);
    LogicalResult initialize(Operation* top) override;
    LogicalResult initializeBlock(Operation* block);
    LogicalResult initializeOperation(Operation* op);
    LogicalResult visit(ProgramPoint* point) override;
};

}
#endif
