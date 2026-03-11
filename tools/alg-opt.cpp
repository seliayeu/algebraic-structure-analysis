#include "Dialect/DIA/DIADialect.h"
#include "Transform/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::bpa::dia::DIADialect>();
    mlir::registerAllPasses();
    mlir::registerAllDialects(registry);

    mlir::bpa::registerBandedPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Algebraic Structure Rewrite Pass Driver", registry));
}
