"""Task 2 experiments: camera convention + TSDF hyper-parameter sweep.

Reconstructs the supplied multi-view RGB-D scene with Open3D's
ScalableTSDFVolume under several settings, computes agreement against the
provided reference mesh, and renders headless preview figures for the report.

Outputs (written under --output_dir):
    task2_metrics.json             all settings and metrics
    mesh_wrong.ply / mesh_correct.ply
    preview_wrong.png / preview_correct.png / preview_target.png
    preview_views.png              view-count sweep
    preview_voxel.png              voxel-size sweep
    preview_sdf.png                sdf_trunc sweep
    preview_depth_trunc.png        depth_trunc sweep
    comparison_chamfer.png         metric curves

Usage:
    conda activate module2
    python task2_experiments.py --data_root . --output_dir task2_results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch  # import first: Windows DLL order workaround used elsewhere in this project
from PIL import Image
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAM_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--points", type=int, default=60000, help="points used for Chamfer approximation")
    p.add_argument("--preview_points", type=int, default=30000)
    return p.parse_args()


def read_rgbd_and_pose(data_root: Path, flip: bool):
    meta = json.loads((data_root / "camera.json").read_text(encoding="utf-8"))
    K = np.asarray(meta["K"], dtype=np.float64)
    K4 = np.eye(4, dtype=np.float64)
    K4[:3, :3] = K
    keys = sorted(k for k in meta if k != "K")

    poses, intrinsics, rgbs, depths = [], [], [], []
    for key in keys:
        pose = np.asarray(meta[key], dtype=np.float64)
        if flip:
            pose = pose @ CAM_FLIP
        rgb = np.asarray(Image.open(data_root / "color" / f"{key}.png").convert("RGB"))
        depth = np.asarray(Image.open(data_root / "depth" / f"{key}.png")).astype(np.float32) / 1000.0
        poses.append(pose)
        intrinsics.append(K4.copy())
        rgbs.append(rgb)
        depths.append(depth)
    return poses, intrinsics, rgbs, depths


def reconstruct(poses, intrinsics, rgbs, depths, view_count: int,
                voxel_size: float, sdf_trunc: float, depth_trunc: float):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    indices = np.linspace(0, len(poses) - 1, int(view_count), dtype=int)
    for i in indices:
        K = intrinsics[i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=rgbs[i].shape[1], height=rgbs[i].shape[0],
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgbs[i], dtype=np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(depths[i], dtype=np.float32)),
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[i]))
    return volume.extract_triangle_mesh()


def mesh_metrics(mesh, ref, n_points: int) -> Dict[str, object]:
    if mesh.is_empty():
        return {"empty": True}
    src = mesh.sample_points_uniformly(number_of_points=n_points)
    tgt = ref.sample_points_uniformly(number_of_points=n_points)
    d1 = np.asarray(src.compute_point_cloud_distance(tgt))
    d2 = np.asarray(tgt.compute_point_cloud_distance(src))
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    return {
        "empty": False,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "chamfer_mm": float((d1.mean() + d2.mean()) * 500.0),
        "mean_src_to_ref_mm": float(d1.mean() * 1000.0),
        "mean_ref_to_src_mm": float(d2.mean() * 1000.0),
        "precision_2mm_pct": float((d1 <= 0.002).mean() * 100.0),
        "recall_2mm_pct": float((d2 <= 0.002).mean() * 100.0),
        "aabb_m": [round(float(v), 4) for v in extent],
    }


def center_scale(mesh):
    points = np.asarray(mesh.vertices)
    center = points.mean(axis=0)
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    scale = float(np.linalg.norm(extent))
    return center, max(scale, 1e-9)


def headless_preview(meshes: List[o3d.geometry.TriangleMesh],
                     titles: List[str], path: Path, elev: float = 22.0,
                     azim: float = 35.0) -> None:
    """Orthographic, colour-preserving preview without opening a GUI window."""
    ref = meshes[0]
    center, scale = center_scale(ref)
    fig = plt.figure(figsize=(4.2 * len(meshes), 4.4))
    for i, (mesh, title) in enumerate(zip(meshes, titles), start=1):
        ax = fig.add_subplot(1, len(meshes), i, projection="3d")
        if mesh.is_empty() or len(mesh.triangles) == 0:
            ax.text2D(0.5, 0.5, "empty mesh", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            ax.set_title(title, fontsize=11)
            continue
        pts = mesh.sample_points_uniformly(number_of_points=18000)
        xyz = np.asarray(pts.points) - center
        rgb = np.asarray(pts.colors)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=0.15, linewidths=0, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        lim = scale * 0.6
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    out = (args.output_dir or root / "task2_results").resolve()
    out.mkdir(parents=True, exist_ok=True)

    ref = o3d.io.read_triangle_mesh(str(root / "recon_tgt.ply"))
    if ref.is_empty():
        raise RuntimeError("recon_tgt.ply could not be loaded")
    target_center, _ = center_scale(ref)

    results: Dict[str, dict] = {"data_root": str(root), "reference": str(root / "recon_tgt.ply"), "experiments": {}}

    # ---------- 1. camera convention ----------
    for flip, name in ((False, "wrong_no_camera_flip"), (True, "correct_camera_flip")):
        poses, intrinsics, rgbs, depths = read_rgbd_and_pose(root, flip=flip)
        # The no-flip fusion lands in a much larger, meaningless volume; a coarse
        # voxel keeps the artefact small enough to store and render.
        voxel = 0.004 if flip else 0.02
        t0 = time.perf_counter()
        mesh = reconstruct(poses, intrinsics, rgbs, depths,
                           view_count=100, voxel_size=voxel, sdf_trunc=0.02, depth_trunc=3.0)
        dt = time.perf_counter() - t0
        m = mesh_metrics(mesh, ref, args.points)
        m.update(seconds=round(dt, 3), views=100, voxel_size=voxel, sdf_trunc=0.02, depth_trunc=3.0,
                 camera_flip=flip)
        results["experiments"][name] = m
        if not mesh.is_empty():
            o3d.io.write_triangle_mesh(str(out / ("mesh_wrong.ply" if not flip else "mesh_correct.ply")), mesh)
            if flip:
                headless_preview([mesh, ref], ["Reconstruction (flipped pose)", "Reference mesh"],
                                 out / "preview_correct.png")
            else:
                headless_preview([mesh, ref], ["Reconstruction (no flip)", "Reference mesh"],
                                 out / "preview_wrong.png")
        print(name, m, flush=True)

    headless_preview([ref], ["Reference mesh"], out / "preview_target.png")

    # ---------- 2. view-count sweep ----------
    poses, intrinsics, rgbs, depths = read_rgbd_and_pose(root, flip=True)
    view_meshes, view_titles = [], []
    for views in [5, 10, 25, 50, 100]:
        t0 = time.perf_counter()
        mesh = reconstruct(poses, intrinsics, rgbs, depths, views, 0.004, 0.02, 3.0)
        m = mesh_metrics(mesh, ref, args.points)
        m.update(seconds=round(time.perf_counter() - t0, 3), views=views)
        results["experiments"][f"views_{views}"] = m
        view_meshes.append(mesh); view_titles.append(f"{views} views")
        print("views", views, m, flush=True)
    headless_preview(view_meshes, view_titles, out / "preview_views.png")

    # ---------- 3. voxel-size sweep ----------
    voxel_meshes, voxel_titles = [], []
    for voxel in [0.002, 0.004, 0.008, 0.016]:
        t0 = time.perf_counter()
        mesh = reconstruct(poses, intrinsics, rgbs, depths, 100, voxel, 0.02, 3.0)
        m = mesh_metrics(mesh, ref, args.points)
        m.update(seconds=round(time.perf_counter() - t0, 3), voxel_size=voxel)
        results["experiments"][f"voxel_{voxel}"] = m
        voxel_meshes.append(mesh); voxel_titles.append(f"voxel={voxel} m")
        print("voxel", voxel, m, flush=True)
    headless_preview(voxel_meshes, voxel_titles, out / "preview_voxel.png")

    # ---------- 4. sdf_trunc sweep ----------
    sdf_meshes, sdf_titles = [], []
    for sdf in [0.005, 0.01, 0.02, 0.05]:
        t0 = time.perf_counter()
        mesh = reconstruct(poses, intrinsics, rgbs, depths, 100, 0.004, sdf, 3.0)
        m = mesh_metrics(mesh, ref, args.points)
        m.update(seconds=round(time.perf_counter() - t0, 3), sdf_trunc=sdf)
        results["experiments"][f"sdf_{sdf}"] = m
        sdf_meshes.append(mesh); sdf_titles.append(f"sdf_trunc={sdf} m")
        print("sdf", sdf, m, flush=True)
    headless_preview(sdf_meshes, sdf_titles, out / "preview_sdf.png")

    # ---------- 5. depth_trunc sweep ----------
    depth_meshes, depth_titles = [], []
    for trunc in [1.0, 2.0, 3.0, 5.0]:
        t0 = time.perf_counter()
        mesh = reconstruct(poses, intrinsics, rgbs, depths, 100, 0.004, 0.02, trunc)
        m = mesh_metrics(mesh, ref, args.points)
        m.update(seconds=round(time.perf_counter() - t0, 3), depth_trunc=trunc)
        results["experiments"][f"depth_{trunc}"] = m
        depth_meshes.append(mesh); depth_titles.append(f"depth_trunc={trunc} m")
        print("depth_trunc", trunc, m, flush=True)
    headless_preview(depth_meshes, depth_titles, out / "preview_depth_trunc.png")

    (out / "task2_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out / "task2_metrics.json")


if __name__ == "__main__":
    main()