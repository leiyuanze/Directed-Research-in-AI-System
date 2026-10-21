"""Compute Task 2 metrics from already-saved meshes and finish the depth sweep."""
import json, time
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task2_experiments import read_rgbd_and_pose, reconstruct, mesh_metrics, headless_preview

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'task2_results'
ref = o3d.io.read_triangle_mesh(str(ROOT / 'recon_tgt.ply'))
N = 50000

def load(name):
    return o3d.io.read_triangle_mesh(str(OUT / name))

results = {'data_root': str(ROOT), 'reference': str(ROOT / 'recon_tgt.ply'), 'experiments': {}}
wrong = load('mesh_wrong.ply'); correct = load('mesh_correct.ply')
m = mesh_metrics(wrong, ref, N); m.update(views=100, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3.0, camera_flip=False, seconds=106.501)
results['experiments']['wrong_no_camera_flip'] = m
m = mesh_metrics(correct, ref, N); m.update(views=100, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3.0, camera_flip=True, seconds=4.288)
results['experiments']['correct_camera_flip'] = m
for v in [5,10,25,50,100]:
    src = ROOT.parent.parent.parent / '.deps' / f'task2_v{v}.ply'
    mm = mesh_metrics(o3d.io.read_triangle_mesh(str(src)), ref, N) if src.exists() else {'missing': str(src)}
    mm['views'] = v
    results['experiments'][f'views_{v}'] = mm
print('views done', flush=True)

poses, intrinsics, rgbs, depths = read_rgbd_and_pose(ROOT, True)
meshes=[]; titles=[]
for trunc in [1.0,2.0,3.0,5.0]:
    t0=time.perf_counter()
    mesh=reconstruct(poses,intrinsics,rgbs,depths,100,0.004,0.02,trunc)
    mm=mesh_metrics(mesh,ref,N); mm.update(seconds=round(time.perf_counter()-t0,3),depth_trunc=trunc)
    results['experiments'][f'depth_{trunc}']=mm
    meshes.append(mesh); titles.append(f'depth_trunc={trunc} m')
    print('depth_trunc', trunc, mm, flush=True)
headless_preview(meshes,titles,OUT/'preview_depth_trunc.png')
(OUT/'task2_metrics_partial.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print('wrote', OUT/'task2_metrics_partial.json')