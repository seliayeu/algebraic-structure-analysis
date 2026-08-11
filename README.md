## Bandedness Propagation Analysis of Tensor Compilers

This repository implements the Bandedness Propagation Analysis (BPA) to propagate bandwidth information across computational graphs. The analysis is implemented in MLIR using the `linalg` dialect and a custom `DIA` dialect to compress banded matrices.

The analysis is implemented as an MLIR pass that annotates operators with bandwidth information, which is then used by a custom code generator to produce banded-aware loop nests that only compute within the necessary bands proved by the analysis.


### Dependencies
BPA is built with MLIR 23 and Clang 17. The build uses CMake 4.4.0 and Ninja 1.13.2.


### Artifact

The available `Dockerfile.artifact` will build the project and generate the figures available in the paper.

Build the Docker image:

```bash
docker build -f Dockerfile.artifact -t bpa-artifact .
```

For ARM based architectures:

```bash
docker build --platform linux/amd64 -f Dockerfile.artifact -t bpa-artifact .
```

Run the container in detached mode:
```bash
docker run -d -v $(pwd)/results:/app/bpa/results --name bpa-experiments bpa-artifact:latest
```

### How to build

1. Clone BPA

$ git clone https://github.com/seliayeu/algebraic-structure-analysis

2. Build

$ mkdir build && cd build
$ cmake -G Ninja ../ && ninja

### Running tests

Once built, you can run the tests inside the build folder:

$ ninja check-bpa

###### Currently supported operators:

- `linalg.matmul`
- `linalg.batch_matmul`
- `linalg.elementwise`
- `linalg.transposition`

- `dia.matmul`
- `dia.batch_matmul`
- `dia.transposition`
- `dia.elementwise`

