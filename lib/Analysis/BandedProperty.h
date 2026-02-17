#ifndef LIB_ANALYSIS_BANDEDPROPERTY_H
#define LIB_ANALYSIS_BANDEDPROPERTY_H

#include <algorithm>
#include <limits>

namespace mlir::bpa {

struct BandedProperty {
    long UpperBandwidth;
    long LowerBandwidth;

    explicit BandedProperty(long upperBandwidth, long lowerBandwidth)
        : UpperBandwidth{ upperBandwidth }, LowerBandwidth{ lowerBandwidth } {
    }

    bool operator==(const BandedProperty& other) const {
        return UpperBandwidth == other.UpperBandwidth && LowerBandwidth == other.LowerBandwidth;
    }
};

static BandedProperty meet(const BandedProperty& lhs, const BandedProperty& rhs) {
    return BandedProperty(std::max(lhs.UpperBandwidth, rhs.UpperBandwidth),
                          std::max(lhs.LowerBandwidth, rhs.LowerBandwidth));
}

static BandedProperty join(const BandedProperty& lhs, const BandedProperty& rhs) {
    return BandedProperty(std::min(lhs.UpperBandwidth, rhs.UpperBandwidth),
                          std::min(lhs.LowerBandwidth, rhs.LowerBandwidth));
}

static BandedProperty binaryMatmul(const BandedProperty& lhs, const BandedProperty& rhs) {
    unsigned long upperBw{ static_cast<unsigned long>(lhs.UpperBandwidth) + rhs.UpperBandwidth };
    upperBw = std::min(upperBw, static_cast<unsigned long>(std::numeric_limits<long>::max()));
    unsigned long lowerBw{ static_cast<unsigned long>(lhs.LowerBandwidth) + rhs.LowerBandwidth };
    lowerBw = std::min(lowerBw, static_cast<unsigned long>(std::numeric_limits<long>::max()));
    return BandedProperty(upperBw, lowerBw);
}

static BandedProperty binaryElementwiseGeneral(const BandedProperty& lhs,
                                               const BandedProperty& rhs) {
    return meet(lhs, rhs);
}

static BandedProperty binaryElementwiseProduct(const BandedProperty& lhs,
                                               const BandedProperty& rhs) {
    return join(lhs, rhs);
}

static BandedProperty unaryElementwise(const BandedProperty& property) {
    return property;
}

static BandedProperty transpose(const BandedProperty& property) {
    return BandedProperty(property.LowerBandwidth, property.UpperBandwidth);
}

}  // namespace mlir::bpa
#endif
