"""Thí nghiệm Benchmark đo đạc độ nhạy tham số cho Topic D (Robot Obstacle Detection).

Tuân thủ nghiêm ngặt 4 nguyên tắc thí nghiệm:
  1. Ít nhất 3 mức cho mỗi tham số (có mốc cơ sở 0 hoặc baseline để so sánh).
  2. Mỗi lần chỉ thay đổi duy nhất một yếu tố (giữ nguyên frame, seed, và các tham số khác).
  3. Tái lập được 100%: cố định random seed để mọi lần chạy ra kết quả giống hệt nhau.
  4. Đo latency chuẩn: đo bằng time.perf_counter(), bỏ lần chạy đầu (warmup),
     chạy lặp lại ít nhất 20 lần và tính trung vị (p50) cùng phân vị 95 (p95).

Cách chạy:
  python -m src.sweep_experiment --data-root data/kitti_mini --frame 000011
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

from starter.datasets import load_frame


def filter_roi(points: np.ndarray,
               x_range: tuple[float, float] = (0.0, 45.0),
               y_range: tuple[float, float] = (-15.0, 15.0),
               z_range: tuple[float, float] = (-2.5, 2.0)) -> np.ndarray:
    """Lọc vùng quan tâm (ROI) phía trước robot."""
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]


def run_single_pipeline(points_roi: np.ndarray,
                        voxel_size: float = 0.1,
                        distance_threshold: float = 0.2,
                        eps: float = 0.5,
                        min_points: int = 10,
                        seed: int = 42) -> tuple[int, int, int, float | None]:
    """Chạy 1 lần pipeline với seed cố định.

    Trả về: (số điểm ground, số điểm obstacle, số cụm clusters, khoảng cách d_min)
    """
    o3d.utility.random.seed(seed)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_roi)

    # 1. Voxel Downsample
    if voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_down = pcd

    down_pts = np.asarray(pcd_down.points)
    if len(down_pts) < 3:
        return 0, 0, 0, None

    # 2. RANSAC Ground Removal
    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=100,
    )
    ground_pcd = pcd_down.select_by_index(inliers)
    obstacle_pcd = pcd_down.select_by_index(inliers, invert=True)

    obstacle_pts = np.asarray(obstacle_pcd.points)
    n_ground = len(inliers)
    n_obstacle = len(obstacle_pts)

    # 3. DBSCAN Clustering
    if len(obstacle_pts) >= min_points:
        labels = np.array(obstacle_pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=False
        ))
    else:
        labels = np.array([-1] * len(obstacle_pts))

    unique_labels = set(labels) - {-1}
    n_clusters = len(unique_labels)

    # 4. Tính khoảng cách tới cụm gần nhất
    min_dist = float("inf")
    for lbl in unique_labels:
        c_pts = obstacle_pts[labels == lbl]
        dists = np.linalg.norm(c_pts[:, :2], axis=1)
        c_min = float(dists.min())
        if c_min < min_dist:
            min_dist = c_min

    d_min = min_dist if min_dist != float("inf") else None
    return n_ground, n_obstacle, n_clusters, d_min


def benchmark_configuration(points_roi: np.ndarray,
                            voxel_size: float,
                            distance_threshold: float,
                            eps: float,
                            min_points: int = 10,
                            n_runs: int = 20,
                            seed: int = 42) -> dict:
    """Đo số liệu và độ trễ p50/p95 cho một cấu hình."""
    # Chạy lần đầu warmup (bỏ qua không đo thời gian)
    n_ground, n_obstacle, n_clusters, d_min = run_single_pipeline(
        points_roi, voxel_size, distance_threshold, eps, min_points, seed=seed
    )

    # Chạy lặp lại n_runs lần để đo độ trễ chuẩn xác
    latencies = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        run_single_pipeline(points_roi, voxel_size, distance_threshold, eps, min_points, seed=seed)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    return {
        "voxel_size_m": voxel_size,
        "distance_threshold_m": distance_threshold,
        "eps_m": eps,
        "min_points": min_points,
        "ground_points": n_ground,
        "obstacle_points": n_obstacle,
        "n_clusters": n_clusters,
        "d_min_m": round(d_min, 2) if d_min is not None else -1.0,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm Sweep Benchmark cho Topic D")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Thư mục dữ liệu KITTI")
    parser.add_argument("--frame", default="000011", help="ID frame cần quét (mặc định: 000011)")
    parser.add_argument("--n-runs", type=int, default=20, help="Số lần chạy lặp để đo latency (mặc định: 20)")
    parser.add_argument("--out-csv", default="results/obstacle_benchmark.csv", help="Đường dẫn lưu CSV")
    parser.add_argument("--out-fig", default="results/figures/obstacle_sweep.png", help="Đường dẫn lưu biểu đồ PNG")
    args = parser.parse_args()

    # Load dữ liệu frame và lọc ROI
    fr = load_frame(args.data_root, args.frame)
    roi_pts = filter_roi(fr["points"][:, :3])
    print(f"=== BẮT ĐẦU THÍ NGHIỆM SWEEP BENCHMARK (Frame {args.frame}) ===")
    print(f"Tổng số điểm trong vùng quan sát (ROI): {len(roi_pts):,}")
    print(f"Số lần lặp đo latency: {args.n_runs} lần/cấu hình (bỏ lần đầu warmup)\n")

    results = []

    # -------------------------------------------------------------
    # THÍ NGHIỆM 1: Quét tham số eps của DBSCAN (0.3 / 0.5 / 0.8 m)
    # Giữ nguyên voxel_size = 0.1 m và distance_threshold = 0.2 m
    # -------------------------------------------------------------
    base_voxel = 0.1
    base_dist = 0.2
    eps_values = [0.3, 0.5, 0.8]

    print("--- 1. Quét tham số DBSCAN eps (0.3 / 0.5 / 0.8 m) ---")
    print(f"{'eps (m)':<10} | {'Số cụm':<10} | {'d_min (m)':<12} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | Nhận xét")
    print("-" * 75)

    for eps in eps_values:
        res = benchmark_configuration(
            roi_pts,
            voxel_size=base_voxel,
            distance_threshold=base_dist,
            eps=eps,
            n_runs=args.n_runs
        )
        note = "Over-segmentation (xé nhỏ)" if eps == 0.3 else (
            "Baseline chuẩn (tách đều)" if eps == 0.5 else "Under-segmentation (dính chùm)"
        )
        print(f"{eps:<10.1f} | {res['n_clusters']:<10} | {res['d_min_m']:<12.2f} | {res['latency_p50_ms']:<10.2f} | {res['latency_p95_ms']:<10.2f} | {note}")

        row = {
            "dataset": "kitti_mini",
            "frame_id": args.frame,
            "experiment_type": "sweep_dbscan_eps",
            "swept_parameter": "eps_m",
            "swept_value": eps,
            **res
        }
        results.append(row)

    # -------------------------------------------------------------
    # THÍ NGHIỆM 2: Quét tham số distance_threshold của RANSAC
    # Giữ nguyên voxel_size = 0.1 m và eps = 0.5 m
    # -------------------------------------------------------------
    dist_values = [0.10, 0.15, 0.20, 0.30, 0.45]
    print("\n--- 2. Quét tham số RANSAC distance_threshold (0.10 / 0.15 / 0.20 / 0.30 / 0.45 m) ---")
    print(f"{'d_th (m)':<10} | {'Điểm đất':<10} | {'Điểm vật cản':<14} | {'Số cụm':<10} | {'p50 (ms)':<10} | Nhận xét")
    print("-" * 85)

    for d_th in dist_values:
        res = benchmark_configuration(
            roi_pts,
            voxel_size=base_voxel,
            distance_threshold=d_th,
            eps=0.5,
            n_runs=args.n_runs
        )
        note = "Sót nhiễu mặt đường" if d_th <= 0.10 else (
            "Baseline chuẩn" if d_th == 0.20 else (
                "Bắt đầu lẹm chân người" if d_th == 0.30 else "Lẹm nặng chân/thân người (<0.45m)"
            )
        )
        print(f"{d_th:<10.2f} | {res['ground_points']:<10} | {res['obstacle_points']:<14} | {res['n_clusters']:<10} | {res['latency_p50_ms']:<10.2f} | {note}")

        row = {
            "dataset": "kitti_mini",
            "frame_id": args.frame,
            "experiment_type": "sweep_ransac_distance_threshold",
            "swept_parameter": "distance_threshold_m",
            "swept_value": d_th,
            **res
        }
        results.append(row)

    # Tạo DataFrame và lưu CSV
    df = pd.DataFrame(results)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)
    print(f"\n-> Đã lưu bảng số liệu vào: {out_csv}")

    # Vẽ biểu đồ 2 panel trực quan
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: eps vs Số cụm & Latency p50
    df_eps = df[df["experiment_type"] == "sweep_dbscan_eps"]
    ax1 = axes[0]
    color1 = "tab:blue"
    ax1.set_xlabel("Bán kính lân cận DBSCAN eps (m)", fontsize=11)
    ax1.set_ylabel("Số cụm vật cản (Clusters)", color=color1, fontsize=11)
    line1 = ax1.plot(df_eps["swept_value"], df_eps["n_clusters"], marker="o", color=color1, linewidth=2, label="Số cụm")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    color2 = "tab:red"
    ax1_twin.set_ylabel("Độ trễ trung vị p50 (ms)", color=color2, fontsize=11)
    line2 = ax1_twin.plot(df_eps["swept_value"], df_eps["latency_p50_ms"], marker="s", color=color2, linestyle="--", linewidth=2, label="Latency p50")
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("1. Ảnh hưởng của DBSCAN eps", fontsize=12, weight="bold")

    # Panel 2: distance_threshold vs Số điểm vật cản & Điểm mặt đất
    df_dist = df[df["experiment_type"] == "sweep_ransac_distance_threshold"]
    ax2 = axes[1]
    ax2.plot(df_dist["swept_value"], df_dist["ground_points"], marker="^", color="forestgreen", linewidth=2, label="Điểm mặt đất (Ground)")
    ax2.plot(df_dist["swept_value"], df_dist["obstacle_points"], marker="v", color="crimson", linewidth=2, label="Điểm vật cản (Obstacles)")
    ax2.set_xlabel("Ngưỡng khoảng cách RANSAC distance_threshold (m)", fontsize=11)
    ax2.set_ylabel("Số lượng điểm", fontsize=11)
    ax2.set_title("2. Ảnh hưởng của RANSAC distance_threshold", fontsize=12, weight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.suptitle(f"Topic D Sweep Benchmark — Frame {args.frame} (KITTI)", fontsize=13, weight="bold")
    plt.tight_layout()

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_fig), dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ vào: {out_fig}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
