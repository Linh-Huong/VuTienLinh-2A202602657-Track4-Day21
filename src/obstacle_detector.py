"""Pipeline phát hiện vật cản cho Robot/Drone từ Point Cloud (Topic D).

Các bước:
  1. Lọc vùng quan tâm (ROI).
  2. Voxel downsample (Open3D).
  3. Tách mặt đất bằng RANSAC plane (Open3D).
  4. Gom cụm vật cản bằng DBSCAN (Open3D).
  5. Trích xuất bounding box và tính khoảng cách vật cản gần nhất.

Cách chạy:
  python -m src.obstacle_detector --data-root data/kitti_mini --frame 000011
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from starter.datasets import load_frame


def filter_roi(points: np.ndarray,
               x_range: tuple[float, float] = (0.0, 45.0),
               y_range: tuple[float, float] = (-15.0, 15.0),
               z_range: tuple[float, float] = (-2.5, 2.0)) -> np.ndarray:
    """Lọc các điểm LiDAR nằm trong vùng không gian quan tâm (ROI) phía trước robot."""
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]


def detect_obstacles(points_xyz: np.ndarray,
                     voxel_size: float = 0.1,
                     distance_threshold: float = 0.2,
                     ransac_n: int = 3,
                     num_iterations: int = 100,
                     eps: float = 0.5,
                     min_points: int = 10) -> dict:
    """Thực hiện trọn vẹn pipeline phát hiện vật cản trên mảng điểm (N, 3)."""
    t0 = time.perf_counter()

    # 1. Chuyển sang Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)

    # 2. Voxel Downsample
    if voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_down = pcd

    down_pts = np.asarray(pcd_down.points)
    if len(down_pts) < ransac_n:
        return {
            "pcd_down": down_pts,
            "ground_pts": np.empty((0, 3)),
            "obstacle_pts": np.empty((0, 3)),
            "labels": np.empty((0,), dtype=int),
            "boxes": [],
            "d_min": None,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # 3. Ground removal bằng RANSAC plane
    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    ground_pcd = pcd_down.select_by_index(inliers)
    obstacle_pcd = pcd_down.select_by_index(inliers, invert=True)

    ground_pts = np.asarray(ground_pcd.points)
    obstacle_pts = np.asarray(obstacle_pcd.points)

    # 4. Clustering bằng DBSCAN
    if len(obstacle_pts) >= min_points:
        labels = np.array(obstacle_pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=False
        ))
    else:
        labels = np.array([-1] * len(obstacle_pts))

    # 5. Trích xuất Bounding Box cho từng cluster
    unique_labels = set(labels) - {-1}
    boxes = []
    min_dist = float("inf")

    for lbl in unique_labels:
        cluster_mask = (labels == lbl)
        cluster_pts = obstacle_pts[cluster_mask]
        min_xyz = cluster_pts.min(axis=0)
        max_xyz = cluster_pts.max(axis=0)

        # Khoảng cách mặt phẳng ngang (x, y) từ robot (0, 0) đến vật cản
        dists = np.linalg.norm(cluster_pts[:, :2], axis=1)
        cluster_min_d = float(dists.min())
        if cluster_min_d < min_dist:
            min_dist = cluster_min_d

        boxes.append({
            "label": int(lbl),
            "n_points": int(cluster_mask.sum()),
            "min_xyz": min_xyz,
            "max_xyz": max_xyz,
            "min_dist": cluster_min_d,
        })

    d_min = min_dist if min_dist != float("inf") else None
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "pcd_down": down_pts,
        "ground_pts": ground_pts,
        "obstacle_pts": obstacle_pts,
        "labels": labels,
        "boxes": boxes,
        "d_min": d_min,
        "plane_model": plane_model,
        "latency_ms": latency_ms,
    }


def visualize_result(raw_points: np.ndarray, res: dict, out_path: Path, frame_id: str) -> None:
    """Vẽ biểu đồ so sánh BEV (Bird's Eye View) và lưu file ảnh kết quả."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # Subplot 1: Điểm sau lọc ROI
    ax1 = axes[0]
    ax1.scatter(raw_points[:, 1], raw_points[:, 0], s=0.5, c=raw_points[:, 2], cmap="viridis")
    ax1.set_title(f"1. Raw ROI ({len(raw_points):,} points)")
    ax1.set_xlabel("Y (Trái/Phải - m)")
    ax1.set_ylabel("X (Trước mặt robot - m)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Tách mặt đất (RANSAC Ground Removal)
    ax2 = axes[1]
    g_pts = res["ground_pts"]
    o_pts = res["obstacle_pts"]
    if len(g_pts) > 0:
        ax2.scatter(g_pts[:, 1], g_pts[:, 0], s=0.5, c="gray", alpha=0.3, label=f"Mặt đất ({len(g_pts):,})")
    if len(o_pts) > 0:
        ax2.scatter(o_pts[:, 1], o_pts[:, 0], s=1.0, c="crimson", label=f"Vật cản ({len(o_pts):,})")
    ax2.set_title(f"2. Tách mặt đất (RANSAC Plane)")
    ax2.set_xlabel("Y (m)")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Gom cụm DBSCAN và Bounding Box
    ax3 = axes[2]
    labels = res["labels"]
    boxes = res["boxes"]
    if len(o_pts) > 0:
        noise = (labels == -1)
        if np.any(noise):
            ax3.scatter(o_pts[noise, 1], o_pts[noise, 0], s=1.0, c="gray", alpha=0.5, label="Nhiễu")

        # Tô màu từng cluster
        clustered = ~noise
        if np.any(clustered):
            scatter = ax3.scatter(o_pts[clustered, 1], o_pts[clustered, 0],
                                  s=2.0, c=labels[clustered], cmap="tab20")

    # Vẽ 2D Bounding Box (BEV) quanh các cụm
    for b in boxes:
        min_x, max_x = b["min_xyz"][0], b["max_xyz"][0]
        min_y, max_y = b["min_xyz"][1], b["max_xyz"][1]
        rect_y = min_y
        rect_x = min_x
        width_y = max_y - min_y
        height_x = max_x - min_x
        rect = plt.Rectangle((rect_y, rect_x), width_y, height_x,
                             fill=False, edgecolor="blue", linewidth=1.2)
        ax3.add_patch(rect)
        ax3.text(rect_y, rect_x + height_x + 0.5, f"{b['min_dist']:.1f}m",
                 color="darkblue", fontsize=8, weight="bold")

    d_min_str = f"{res['d_min']:.2f}m" if res['d_min'] is not None else "N/A"
    ax3.set_title(f"3. DBSCAN ({len(boxes)} clusters) | d_min={d_min_str}")
    ax3.set_xlabel("Y (m)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    for ax in axes:
        ax.set_xlim([-15, 15])
        ax.set_ylim([0, 45])
        # Vẽ vị trí Robot tại gốc (0, 0)
        ax.plot(0, 0, marker="^", color="black", markersize=10)

    fig.suptitle(f"Topic D: Robot Obstacle Detection — Frame {frame_id} (Thời gian xử lý: {res['latency_ms']:.1f} ms)",
                 fontsize=14, weight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phát hiện vật cản cho robot từ point cloud (RANSAC + DBSCAN)")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Đường dẫn bộ dữ liệu KITTI")
    parser.add_argument("--frame", default="000011", help="ID frame cần chạy, ví dụ: 000011")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Kích thước voxel downsample (m)")
    parser.add_argument("--distance-threshold", type=float, default=0.2, help="Ngưỡng khoảng cách RANSAC plane (m)")
    parser.add_argument("--eps", type=float, default=0.5, help="Bán kính lân cận DBSCAN (m)")
    parser.add_argument("--min-points", type=int, default=10, help="Số điểm tối thiểu tạo cụm DBSCAN")
    parser.add_argument("--out-dir", default="results/figures", help="Thư mục lưu ảnh kết quả")
    args = parser.parse_args()

    # 1. Đọc dữ liệu frame
    fr = load_frame(args.data_root, args.frame)
    points = fr["points"][:, :3]

    # 2. Lọc ROI
    roi_points = filter_roi(points)

    # 3. Chạy pipeline phát hiện vật cản
    res = detect_obstacles(
        roi_points,
        voxel_size=args.voxel_size,
        distance_threshold=args.distance_threshold,
        eps=args.eps,
        min_points=args.min_points,
    )

    # 4. Trực quan hoá và lưu ảnh
    out_file = Path(args.out_dir) / f"demo_d_obstacle_{args.frame}.png"
    visualize_result(roi_points, res, out_file, args.frame)

    print(f"=== KẾT QUẢ PHÁT HIỆN VẬT CẢN (Frame {args.frame}) ===")
    print(f"- Điểm ban đầu (ROI): {len(roi_points):,}")
    print(f"- Điểm sau Voxel Downsample: {len(res['pcd_down']):,}")
    print(f"- Điểm mặt đất: {len(res['ground_pts']):,} | Điểm vật cản: {len(res['obstacle_pts']):,}")
    print(f"- Số cụm vật cản (DBSCAN): {len(res['boxes'])}")
    print(f"- Khoảng cách tới vật cản gần nhất (d_min): {res['d_min']:.2f} m" if res['d_min'] else "- Không có vật cản")
    print(f"- Thời gian xử lý: {res['latency_ms']:.1f} ms")
    print(f"-> Đã lưu ảnh demo tại: {out_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
