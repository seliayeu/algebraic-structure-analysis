import os
from pathlib import Path

from lit.formats import ShTest

config.name = "asa_tester"
config.test_format = ShTest()
config.suffixes = [".mlir"]
