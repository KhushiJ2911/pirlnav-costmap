"""
Generate per-frame 3D pointmaps from depth + poses.

Workflow:
1. Resolve an image directory (`images_fov90`, `images`, `images_downsampled_fov120`,
   `images_downsampled`, `rgb`, `color`) and infer frame count from image files.
2. Load camera intrinsics from the first readable image file.
3. Load poses from `poses_odom.txt` (`idx x y z qx qy qz qw`) with fallback
   to `poses.txt`; if no valid text poses are available, fallback to `agent_states.npy`.
4. Select a depth directory (`images_depth`, `images_depth_downsampled_fov120`,
   `images_depth_downsampled`, `depth_raw`) and load per-frame depth maps.
5. Back-project each valid depth pixel into 3D world coordinates using each pose.
6. Populate `H x W x 3` pointmaps and save them as
   `gt_pointmaps_fov90/<frame:05d>.npy` in the session folder.

Required session inputs:
- Images directory (one of the supported names above) with frame files (`.jpg`, `.png`,
  or `.jpeg`), used for frame count and intrinsics.
- Pose source: `poses_odom.txt` or `poses.txt` (preferred), otherwise `agent_states.npy`.
- Depth files in one of the supported depth directories.

Note:
- If both text poses and `agent_states.npy` are missing/invalid, execution fails.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist

# session_basedir = "/scratch2/public_scratch/toponav/indoor-topo-loc/datasets/hm3d_navigation/hm3d_generated/stretch_maps/hm3d_iin_train/bigger_bot_0.3-sh_0.4"
session_basedir = session_basedir = "/home2/khushi.ramdhani/pirlnav/costmap_gen/sessions"
fps_degree = 500  # Points per frame after FPS sampling
covisibility_threshold = 0.05  # Meters
window_size = 3  # check 3 on right

def resolve_images_dir(session_folder):
    for name in (
        "images_fov90",
        "images",
        "images_downsampled_fov120",
        "images_downsampled",
        "rgb",
        "color",
    ):
        candidate = os.path.join(session_folder, name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"No images directory found in {session_folder}")


def count_frames_from_images(images_dir):
    count = 0
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        count += len([p for p in os.listdir(images_dir) if p.endswith(ext[1:])])
    return count


def get_image_size(images_dir):
    for ext in (".jpg", ".png", ".jpeg"):
        for name in sorted(os.listdir(images_dir)):
            if not name.endswith(ext):
                continue
            img = cv2.imread(os.path.join(images_dir, name))
            if img is None:
                continue
            h, w = img.shape[:2]
            return w, h
    raise ValueError(f"No readable images found in {images_dir}")

def _quat_to_xyzw(q):
    if all(hasattr(q, k) for k in ("x", "y", "z", "w")):
        return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
    if hasattr(q, "components"):
        comps = np.asarray(q.components, dtype=np.float32).reshape(-1)
        if comps.shape[0] == 4:
            w, x, y, z = comps
            return np.array([x, y, z, w], dtype=np.float32)
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if q_arr.shape[0] == 4:
        return q_arr
    raise ValueError("Unsupported quaternion format")

def _pose_from_agent_state(state):
    position = np.array(state.position, dtype=np.float32)
    quaternion_xyzw = _quat_to_xyzw(state.rotation)
    return position, quaternion_xyzw

def load_poses(session_folder, num_frames):
    """Load poses from poses text files with a fallback to agent_states.npy.

    Expected format:
        idx x y z qx qy qz qw
    """
    poses = {}
    poses_path = os.path.join(session_folder, "poses_odom.txt")
    fallback_path = os.path.join(session_folder, "poses.txt")
    if not os.path.exists(poses_path):
        poses_path = fallback_path

    if os.path.exists(poses_path):
        with open(poses_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue

                frame_idx = int(parts[0])
                if frame_idx >= num_frames:
                    continue

                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = (
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                )
                poses[frame_idx] = (np.array([x, y, z], dtype=np.float32), np.array([qx, qy, qz, qw], dtype=np.float32))

    if not poses:
        # Fallback to pickled agent state exports.
        states_path = os.path.join(session_folder, "agent_states.npy")
        if not os.path.exists(states_path):
            raise FileNotFoundError(
                f"No pose file found in {session_folder}; expected poses_odom.txt, poses.txt, or agent_states.npy"
            )
        states = np.load(states_path, allow_pickle=True)
        total = min(len(states), num_frames)
        for frame_idx in range(total):
            state = states[frame_idx]
            try:
                position, quaternion_xyzw = _pose_from_agent_state(state)
            except Exception as exc:
                raise ValueError(f"Failed to parse agent_states entry {frame_idx}: {exc}") from exc
            poses[frame_idx] = (position, quaternion_xyzw)
    
    if not poses:
        raise ValueError(f"No valid poses loaded from {poses_path}")

    return poses

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2],
    ])

def get_camera_pose(position, quaternion):
    """
    Convert Habitat pose to camera pose with coordinate system conversion.
    Habitat uses RUB (Right-Up-Back), we need RDF (Right-Down-Forward).
    """
    # Build 4x4 transformation matrix
    R = quaternion_to_rotation_matrix(quaternion)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = (
        np.asarray(position, dtype=np.float32)
            + np.array([0.0, 0.4, 0.0], dtype=np.float32)
)
    
    # Habitat to camera coordinate system conversion (RUB -> RDF)
    habitat_to_camera = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)
    
    # Apply conversion
    camera_pose = transform @ habitat_to_camera
    
    return camera_pose

def backproject_depth_to_3d(depth_map, K, position, quaternion):
    H, W = depth_map.shape
    v, u = np.mgrid[0:H, 0:W]
    
    # Filter valid depth
    valid = (depth_map > 0.1) & (depth_map < 10.0) & np.isfinite(depth_map)
    if not np.any(valid):
        return None
    
    # Get valid pixels and depths
    u_valid = u[valid]
    v_valid = v[valid]
    depth_valid = depth_map[valid]
    
    # Back-project to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u_valid - cx) * depth_valid / fx
    Y = (v_valid - cy) * depth_valid / fy
    Z = depth_valid
    points_cam = np.stack([X, Y, Z], axis=-1)
    
    # Add homogeneous coordinate
    points_cam_homo = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    
    # Get camera pose with coordinate system conversion
    camera_pose = get_camera_pose(position, quaternion)
    
    # Transform to world coordinates
    points_world_homo = (camera_pose @ points_cam_homo.T).T
    points_world = points_world_homo[:, :3]
    
    pixels = np.stack([u_valid, v_valid], axis=-1)
    
    return {
        "points_3d": points_world,
        "pixels": pixels,
        "depths": depth_valid,
    }


def visualize_points_on_image(rgb_image, points_2d, color=(0, 0, 255), radius=1):
    img = rgb_image.copy()
    pts = np.round(points_2d).astype(int)
    h, w = img.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    for (u, v) in pts:
        cv2.circle(img, (u, v), radius=radius, color=color, thickness=-1)
    return img


def process_session(session_folder):
    images_dir = resolve_images_dir(session_folder)
    width, height = get_image_size(images_dir)
    HFOV_DEG = 120.0
    fx = fy = (float(width) / 2.0) / np.tan(np.radians(HFOV_DEG) / 2.0)

    K = np.array(
        [
            [fx, 0.0, float(width) / 2.0],
            [0.0, fy, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
  )
    num_frames = count_frames_from_images(images_dir)
    if num_frames <= 0:
        raise ValueError(f"No images found in {images_dir}")
    # poses = load_poses_strided(session_folder, num_frames, 5)
    poses = load_poses(session_folder, num_frames)
    print(f"Loaded {num_frames} frames")
    print(f"Camera intrinsics: {K}")

    # Cell: Step 1: Back-project depth to 3D

    frames = []
    all_points = []

    # Determine depth directory
    depth_dir_candidates = ["images_depth", "images_depth_fov90", "images_depth_downsampled_fov120", "images_depth_downsampled", "depth_raw"]
    depth_dir = None
    for candidate in depth_dir_candidates:
        test_path = os.path.join(session_folder, candidate)
        if os.path.isdir(test_path):
            depth_dir = test_path
            break

    if depth_dir is None:
        print(f"Warning: No depth directory found in {session_folder}")
        depth_dir = os.path.join(session_folder, "depth_raw")  # Default fallback

    print(f"Using depth directory: {depth_dir}")

    for frame_idx in tqdm(range(num_frames), desc="Back-projecting depth"):
        base_path = os.path.join(depth_dir, f"{frame_idx:05d}")
        depth_png_file = base_path + ".png"
        depth_npy_file = base_path + ".npy"
        
        # Load depth map (try PNG first, then NPY)
        depth_map = None
        if os.path.exists(depth_png_file):
            # PNG in millimeters, convert to meters
            depth_raw = cv2.imread(depth_png_file, cv2.IMREAD_UNCHANGED)
            if depth_raw is not None:
                # Handle multi-channel PNGs - take first channel if needed
                if len(depth_raw.shape) == 3:
                    depth_raw = depth_raw[:, :, 0]
                depth_map = depth_raw.astype(np.float64) * 0.001
        elif os.path.exists(depth_npy_file):
            # NPY already in meters
            try:
                depth_map = np.load(depth_npy_file).astype(np.float64)
            except Exception as e:
                print(f"Warning: Failed to load depth NPY for frame {frame_idx}: {e}")
        
        if depth_map is None:
            frames.append(None)
            all_points.append(None)
            continue
        
        position, quaternion = poses[frame_idx]
        
        result = backproject_depth_to_3d(depth_map, K, position, quaternion)
        
        if result is None:
            frames.append(None)
            all_points.append(None)
            continue
        
        frames.append({
            "frame_idx": frame_idx,
            "position": position,
            "quaternion": quaternion,
            **result,
        })
        
        all_points.append(result["points_3d"])

    print(f"\nProcessed {num_frames} frames")

    valid_points = [p for p in all_points if p is not None]
    if valid_points:
        all_points_stacked = np.vstack(valid_points)
        bounds_min = np.min(all_points_stacked, axis=0)
        bounds_max = np.max(all_points_stacked, axis=0)

        print(f"Bounds: {bounds_min} to {bounds_max}")

        for frame in frames:
            if frame is None:
                continue
            frame["colors"] = (
                (frame["points_3d"] - bounds_min) / (bounds_max - bounds_min + 1e-6)
                * 255
            ).clip(0, 255).astype(np.uint8)
    else:
        print("No valid 3D points found; skipping colorization")

    print("Converted XYZ to RGB colors")

    # Save pointmaps from depth png or npy
    pointmap_dir = os.path.join(session_folder, "gt_pointmaps_fov90")
    os.makedirs(pointmap_dir, exist_ok=True)

    saved = 0
    skipped = 0

    # Determine depth directory
    depth_dir_candidates = ["images_depth", "images_depth_fov90", "images_depth_downsampled_fov120", "images_depth_downsampled", "depth_raw"]
    depth_dir = None
    for candidate in depth_dir_candidates:
        test_path = os.path.join(session_folder, candidate)
        if os.path.isdir(test_path):
            depth_dir = test_path
            break

    if depth_dir is None:
        print(f"Warning: No depth directory found in {session_folder}")
        depth_dir = os.path.join(session_folder, "depth_raw")  # Default fallback

    print(f"Using depth directory: {depth_dir}")

    for frame in tqdm(frames, desc="Saving pointmaps"):
        if frame is None:
            skipped += 1
            continue

        frame_idx = int(frame["frame_idx"])
        base_path = os.path.join(depth_dir, f"{frame_idx:05d}")
        depth_png_file = base_path + ".png"
        depth_npy_file = base_path + ".npy"

        # Load depth map (try PNG first, then NPY)
        depth_meters = None
        if os.path.exists(depth_png_file):
            # PNG in millimeters, convert to meters
            depth_raw = cv2.imread(depth_png_file, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                print(f"Warning: Failed to read depth PNG: {depth_png_file}")
                skipped += 1
                continue
            depth_meters = depth_raw.astype(np.float64) * 0.001
        elif os.path.exists(depth_npy_file):
            # NPY already in meters
            try:
                depth_meters = np.load(depth_npy_file).astype(np.float64)
            except Exception as e:
                print(f"Warning: Failed to load depth NPY: {depth_npy_file} ({e})")
                skipped += 1
                continue
        else:
            # print(f"Warning: No depth file found for frame {frame_idx}")
            skipped += 1
            continue

        H, W = depth_meters.shape
        
        # Create empty pointmap
        pm = np.full((H, W, 3), np.nan, dtype=np.float32)

        pix = np.round(frame["pixels"]).astype(int)
        pts = frame["points_3d"].astype(np.float32)

        # Safety clip (should already be in-bounds)
        pix[:, 0] = np.clip(pix[:, 0], 0, W - 1)
        pix[:, 1] = np.clip(pix[:, 1], 0, H - 1)

        pm[pix[:, 1], pix[:, 0], :] = pts
        out_path = os.path.join(pointmap_dir, f"{frame_idx:05d}.npy")
        np.save(out_path, pm)
        saved += 1

    print(f"Saved {saved} pointmaps to: {pointmap_dir}")
    print(f"Skipped {skipped} frames (no data)")


def main():
    if not os.path.isdir(session_basedir):
        raise ValueError(f"session_basedir is not a directory: {session_basedir}")

    session_dirs = [
        os.path.join(session_basedir, d)
        for d in sorted(os.listdir(session_basedir))
        if os.path.isdir(os.path.join(session_basedir, d))
    ]
    if not session_dirs:
        raise ValueError(f"No session directories found under: {session_basedir}")

    for session_folder in session_dirs:
        print(f"\n=== Processing: {session_folder} ===")
        try:
            process_session(session_folder)
        except Exception as exc:
            print(f"[WARN] Failed {session_folder}: {exc}")

if __name__ == "__main__":
    main()