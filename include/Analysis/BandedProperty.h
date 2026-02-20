#ifndef LIB_ANALYSIS_BANDEDPROPERTY_H
#define LIB_ANALYSIS_BANDEDPROPERTY_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlir::bpa {

struct BandedProperty {
    uint64_t UpperBandwidth;
    uint64_t LowerBandwidth;

    explicit BandedProperty(uint64_t upperBandwidth, uint64_t lowerBandwidth)
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
    uint64_t upperBw{ lhs.UpperBandwidth >
                              (std::numeric_limits<int64_t>::max() - rhs.UpperBandwidth)
                          ? std::numeric_limits<int64_t>::max()
                          : rhs.UpperBandwidth + lhs.UpperBandwidth };
    uint64_t lowerBw{ lhs.LowerBandwidth >
                              (std::numeric_limits<int64_t>::max() - rhs.LowerBandwidth)
                          ? std::numeric_limits<int64_t>::max()
                          : rhs.LowerBandwidth + lhs.LowerBandwidth };
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
