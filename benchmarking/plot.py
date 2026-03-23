import csv
import matplotlib.pyplot as plt


def load_csv(path):
    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["bandwidth"]), float(r["avg_time_s"])))

    # sort by bandwidth decreasing
    rows.sort(key=lambda x: x[0], reverse=False)

    bandwidth = [r[0] for r in rows]
    avg_time = [r[1] for r in rows]

    return bandwidth, avg_time


dense_bw, dense_time = load_csv("benchmarking/kalman_dense.csv")
bpa_bw, bpa_time = load_csv("benchmarking/kalman_analysis.csv")

plt.figure()

plt.plot(dense_bw, dense_time, color="red", marker="o", label="Dense")
plt.plot(bpa_bw, bpa_time, color="green", marker="o", label="Analysis")

plt.xlabel("Bandwidth")
plt.ylabel("Average Time (s)")
plt.title("Kernel Performance vs Bandwidth")

plt.gca().invert_xaxis()

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("benchmarking/kalman_plot.png", dpi=300)
plt.show()
