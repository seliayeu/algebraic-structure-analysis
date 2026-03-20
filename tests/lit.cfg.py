import os

from lit.formats import ShTest

config.name = "bpa_tester"
config.test_format = ShTest()
config.suffixes = [".mlir"]

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
build_dir = os.path.abspath(os.path.join(project_root, "build"))

config.substitutions.append(("%project_root", project_root))
config.substitutions.append(("%build", build_dir))
llvm_root = os.environ.get("LLVM_ROOT", "")
config.substitutions.append(("%llvm_root", llvm_root))
config.substitutions.append(("%lib_format", "so"))
