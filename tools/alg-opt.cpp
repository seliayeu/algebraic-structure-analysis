#include "lib/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::registerAllPasses();
    mlir::bpa::registerBandedStructureDebug();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Algebraic Structure Rewrite Pass Driver", registry));
}
