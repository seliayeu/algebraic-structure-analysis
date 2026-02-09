#ifndef LIB_ANALYSIS_ALGEBRAICSTRUCTUREANALYSIS_H
#define LIB_ANALYSIS_ALGEBRAICSTRUCTUREANALYSIS_H
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include <string>
#include <iostream>
#include <limits>
#include <array>

namespace mlir::asa {

enum class AlgebraicProperty : unsigned int {
    Identity,
    Diagonal,
    Symmetric,
    UpperTriangular,
    LowerTriangular,
    General,
};

struct SubMatrixProperty {
    AlgebraicProperty property;
    std::array<long long, 2> dimensions{ -1, -1 };
    bool operator==(const SubMatrixProperty& other) const {
        return (property == other.property) && (dimensions == other.dimensions);
    }
    bool operator!=(const SubMatrixProperty& other) const {
        return !operator==(other);
    }
};

static AlgebraicProperty meet(const AlgebraicProperty lhs, const AlgebraicProperty rhs) {
    if (lhs == AlgebraicProperty::Identity)
        return rhs;
    if (rhs == AlgebraicProperty::Identity)
        return lhs;
    if (lhs == AlgebraicProperty::Diagonal)
        return rhs;
    if (rhs == AlgebraicProperty::Diagonal)
        return lhs;
    if (lhs == rhs)
        return lhs;
    return AlgebraicProperty::General;
}

static AlgebraicProperty join(const AlgebraicProperty& lhs, const AlgebraicProperty& rhs) {
    // pessimistic join
    if (lhs == AlgebraicProperty::General)
        return rhs;
    if (rhs == AlgebraicProperty::General)
        return lhs;
    if (lhs == AlgebraicProperty::Symmetric || lhs == AlgebraicProperty::UpperTriangular || lhs == AlgebraicProperty::LowerTriangular)
        return AlgebraicProperty::Diagonal; // rhs is either sym, upper, lower, or diagonal. either way, output is diagonal
    if (rhs == AlgebraicProperty::Symmetric || rhs == AlgebraicProperty::UpperTriangular || rhs == AlgebraicProperty::LowerTriangular)
        return AlgebraicProperty::Diagonal; // rhs is either sym, upper, lower, or diagonal. either way, output is diagonal
    if (lhs == AlgebraicProperty::Diagonal)
        return rhs;
    if (rhs == AlgebraicProperty::Diagonal)
        return lhs;
    if (lhs == AlgebraicProperty::Identity)
        return rhs;
    if (rhs == AlgebraicProperty::Identity)
        return lhs;
    return lhs; // identity
}

static std::string propertyToString(AlgebraicProperty property) {
    switch (property) {
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

static const AlgebraicProperty stringToValue(const std::string& value) {
    static const DenseMap<StringRef, AlgebraicProperty> dict {
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

static AlgebraicProperty binaryElementwiseGeneral(AlgebraicProperty lhs, AlgebraicProperty rhs) {
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

static AlgebraicProperty binaryElementwiseProduct(AlgebraicProperty lhs, AlgebraicProperty rhs) {
    return join(lhs, rhs);
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

class AlgebraicStructureAnalysis {
    llvm::DenseMap<mlir::Value, SubMatrixProperty> propertyMap;
public:
    LogicalResult run(Block* block);
    SubMatrixProperty getProperty(mlir::Value value) {
        return propertyMap[value];
    }

    bool hasProperty(mlir::Value value) {
        return propertyMap.contains(value);
    }

private:
    LogicalResult visitOperation(Operation* op);
    SubMatrixProperty readPropertyFromDictAttr(DictionaryAttr dictAttr) {
        auto metadataAttr{ dictAttr.get("metadata") };
        if (!metadataAttr)
            return { AlgebraicProperty::General };
        auto innerDictAttr{ dyn_cast<DictionaryAttr>(metadataAttr) };
        if (!innerDictAttr)
            return { AlgebraicProperty::General };
        auto analysisPropertyAttr{ innerDictAttr.get("analysisState") };
        if (!analysisPropertyAttr)
            return { AlgebraicProperty::General };
        auto analysisProperty{ dyn_cast<StringAttr>(analysisPropertyAttr).getValue() };

        auto propertyDimsAttr{ innerDictAttr.get("propertyDims") };
        if (!propertyDimsAttr)
            return { AlgebraicProperty::General };
        auto propertyDimsArrayAttr{ dyn_cast<ArrayAttr>(propertyDimsAttr) };
        if (!propertyDimsArrayAttr || propertyDimsArrayAttr.size() != 2) {
            return { AlgebraicProperty::General };
        }
        SubMatrixProperty res{ stringToValue(std::string(analysisProperty)) }; 
        res.dimensions[0] = cast<IntegerAttr>(propertyDimsArrayAttr[0]).getInt();
        res.dimensions[1] = cast<IntegerAttr>(propertyDimsArrayAttr[1]).getInt();

        return res;
    }
};
}

#endif
