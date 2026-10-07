"""Vẽ kết quả benchmark Topic D. Chạy từ gốc repo: python -m src.plot_sweep"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

csv_path = Path("results/obstacle_benchmark.csv")
if not csv_path.exists():
    print(f"Lỗi: Không tìm thấy file {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: eps vs Số cụm & Latency p50
df_eps = df[df["experiment_type"] == "sweep_dbscan_eps"]
ax1 = axes[0]
color1 = "tab:blue"
ax1.set_xlabel("Bán kính lân cận DBSCAN eps (m)", fontsize=11)
ax1.set_ylabel("Số cụm vật cản (cụm)", color=color1, fontsize=11)
line1 = ax1.plot(df_eps["swept_value"], df_eps["n_clusters"], marker="o", color=color1, linewidth=2, label="Số cụm")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_ylim(bottom=0)
ax1.grid(alpha=0.3)

ax1_twin = ax1.twinx()
color2 = "tab:red"
ax1_twin.set_ylabel("Độ trễ trung vị p50 (ms)", color=color2, fontsize=11)
line2 = ax1_twin.plot(df_eps["swept_value"], df_eps["latency_p50_ms"], marker="s", color=color2, linestyle="--", linewidth=2, label="Latency p50")
ax1_twin.tick_params(axis="y", labelcolor=color2)
ax1_twin.set_ylim(bottom=0)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right")
ax1.set_title("1. Ảnh hưởng của bán kính DBSCAN eps", fontsize=12, weight="bold")

# Panel 2: distance_threshold vs Số điểm vật cản & Điểm mặt đất
df_dist = df[df["experiment_type"] == "sweep_ransac_distance_threshold"]
ax2 = axes[1]
ax2.plot(df_dist["swept_value"], df_dist["ground_points"], marker="^", color="forestgreen", linewidth=2, label="Điểm mặt đất (Ground)")
ax2.plot(df_dist["swept_value"], df_dist["obstacle_points"], marker="v", color="crimson", linewidth=2, label="Điểm vật cản (Obstacles)")
ax2.set_xlabel("Ngưỡng phẳng RANSAC distance_threshold (m)", fontsize=11)
ax2.set_ylabel("Số lượng điểm (điểm)", fontsize=11)
ax2.set_title("2. Ảnh hưởng của ngưỡng RANSAC distance_threshold", fontsize=12, weight="bold")
ax2.set_ylim(bottom=0)
ax2.grid(alpha=0.3)
ax2.legend()

plt.suptitle("Topic D Sweep Benchmark — Frame 000011 (KITTI)", fontsize=13, weight="bold")
fig.tight_layout()

out = Path("results/figures/obstacle_sweep.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"-> Đã lưu biểu đồ vào: {out}")
