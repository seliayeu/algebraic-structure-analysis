#include "Transform/BandedLoweringPass.h"
#include "Transform/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;

    mlir::bpa::registerBandedStructureDebug();
    mlir::bpa::registerBandedLoweringPass();
    mlir::registerAllPasses();
    mlir::registerAllDialects(registry);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Algebraic Structure Rewrite Pass Driver", registry));
}
