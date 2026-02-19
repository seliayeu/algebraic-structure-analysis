#include "Transform/Banded/BandedLoweringPass.h"
#include "Transform/Banded/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::registerAllPasses();
    mlir::bpa::registerBandedStructureDebug();
    mlir::bpa::registerBandedLoweringPass();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Algebraic Structure Rewrite Pass Driver", registry));
}
