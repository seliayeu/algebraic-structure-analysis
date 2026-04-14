import sys
import time
import numpy as np
from scipy.sparse import dia_matrix

n = 1024


def make_dia_band(n, bw):
    offsets = np.arange(-bw, bw + 1)
    data = np.ones((len(offsets), n), dtype=np.float32)
    return dia_matrix((data, offsets), shape=(n, n))


def run_kernel(bw):
    A = make_dia_band(n, bw)
    P0 = make_dia_band(n, bw)
    Q = make_dia_band(n, bw)

    At = A.transpose()
    R1 = A @ P0
    R2 = R1 @ At
    P1 = R2 + Q

    return P1.data[0]


if __name__ == "__main__":
    bw = int(sys.argv[1])
    warmup = int(sys.argv[2]) if len(sys.argv) == 3 else 1

    for i in range(warmup):
        run_kernel(bw)

    start = time.time()
    result = run_kernel(bw)
    end = time.time()
    print(f"total time: {end - start:.6f}")
    print(f"avg time: {end - start:.6f}")
