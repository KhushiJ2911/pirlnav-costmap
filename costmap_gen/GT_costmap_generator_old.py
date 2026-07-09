"""
Compute geodesic distance costmaps for Habitat-Sim NavMesh.
Optimized with Multiprocessing and "Hole Filling" for disconnected islands.
This script builds per-frame navmesh-aligned geodesic costmaps for a session.

Workflow:
1. Read goal arguments from CLI (`goal_frame`, `goal_pixel_u`, `goal_pixel_v`).
2. Read frame count from images in the session image directory.
3. Load pointmaps from `gt_pointmaps_fov90/` (fallback: `pointmaps/`), each named `%05d.npy`.
4. Create a Habitat simulator, load navmesh from `--navmesh_path`, and snap the goal 3D point from the goal frame pointmap.
5. Optionally load `agent_states.npy` to enable floor-aware snapping.
6. For each valid frame pixel:
   - Snap the pixel's 3D point to navmesh.
   - Query geodesic distance from that point to the goal using `habitat_sim.ShortestPath`.
7. Fill disconnected/invalid pixels with nearest valid-neighbor values (KD-tree in pixel space).
8. Save filled costmaps to `output_dir/arrays/<frame>.npy` and PNG visualizations to `output_dir/<frame>.png`.

Inputs required from the session folder:
- an image directory (`images_fov90`, `images`, `images_downsampled_fov120`, `images_downsampled`, `rgb`, or `color`)
- gt_pointmaps_fov90/00000.npy (or pointmaps/ fallback)
- agent_states.npy (optional; better floor-aware snapping when present)
- --navmesh_path file

Note:
- The script does not auto-generate pointmaps. If neither `gt_pointmaps_fov90/` nor
  `pointmaps/` contains per-frame `.npy` files, execution fails early.

Usage:
    pixi run python scripts/GT_costmap_generator_old.py --session_folder <path> --goal_frame <idx> [options]
    
    pixi run python scripts/GT_costmap_generator_old.py --session_folder /home/onyx/work_dirs/aditya/VGGTNav/data/1W61QJVDBqe --goal_frame 104 --goal_pixel_u 200 --goal_pixel_v 120 --navmesh_path /home/onyx/work_dirs/aditya/VGGTNav/data/1W61QJVDBqe/1W61QJVDBqe.basis.navmesh --output_dir /home/onyx/work_dirs/aditya/VGGTNav/data/costmaps_goal_test

"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Optional, Any

import matplotlib
import numpy as np
import habitat_sim
from scipy.spatial import cKDTree  # Added for fast hole filling
from tqdm import tqdm

# Agg backend for headless plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variables for worker processes
_worker_navmesh: Optional[habitat_sim.PathFinder] = None
_worker_goal_snapped: Optional[np.ndarray] = None
_worker_camera_positions: Optional[dict] = None


def init_worker(navmesh_path: str, goal_snapped: np.ndarray, camera_positions: dict):
    """Initialize the worker process with NavMesh and camera positions."""
    global _worker_navmesh, _worker_goal_snapped, _worker_camera_positions
    
    try:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        config = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(config)
        
        if not sim.pathfinder.load_nav_mesh(navmesh_path):
            raise RuntimeError(f"Worker failed to load navmesh: {navmesh_path}")
            
        _worker_navmesh = sim.pathfinder
        _worker_goal_snapped = goal_snapped
        _worker_camera_positions = camera_positions
        
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise


def snap_point_to_floor(
    point_3d: np.ndarray,
    navmesh: habitat_sim.PathFinder,
    camera_height: float,
    height_tolerance: float = 0.5
) -> np.ndarray:
    """
    Snap a 3D point to the navmesh, preferring the floor at camera level.
    
    For ceiling points (above camera), force snapping downward to avoid
    snapping to upper floors in multi-story buildings.
    
    Args:
        point_3d: 3D point to snap
        navmesh: Habitat PathFinder
        camera_height: Y-coordinate of camera position
        height_tolerance: Vertical tolerance for considering same floor (meters)
        
    Returns:
        Snapped 3D point on the navigable surface
    """
    # Check if point is above camera (likely ceiling)
    is_ceiling = point_3d[1] > camera_height + height_tolerance
    
    if is_ceiling:
        # For ceiling points, create a point below and snap that
        # This forces snapping to the floor below rather than floor above
        point_below = point_3d.copy()
        point_below[1] = camera_height - 0.5  # Force it below camera
        snapped = navmesh.snap_point(point_below)
    else:
        # For floor/wall points, normal snapping is fine
        snapped = navmesh.snap_point(point_3d)
    
    # Verify the snapped point is on the correct floor
    # If it's too far vertically from camera level, try to re-snap
    if abs(snapped[1] - camera_height) > 3.0:  # More than 3m away vertically
        # Try snapping with constrained height
        point_constrained = point_3d.copy()
        point_constrained[1] = camera_height
        snapped = navmesh.snap_point(point_constrained)
    
    return snapped


def fill_holes_nearest(costmap: np.ndarray, invalid_val: float = np.inf) -> np.ndarray:
    """
    Fills 'inf' values (holes) in the costmap using the value of the 
    nearest valid pixel (Euclidean distance in pixel space).
    
    This fixes issues where points snap to disconnected NavMesh islands 
    (like tables) by assigning them the cost of the adjacent floor.
    """
    H, W = costmap.shape
    
    # Identify valid and invalid pixels
    # We consider anything finite as valid.
    valid_mask = np.isfinite(costmap)
    
    # If the whole image is empty or full, return as is
    if valid_mask.all():
        return costmap
    if not valid_mask.any():
        return costmap

    # Get coordinates
    # y_valid, x_valid = (N, ), (N, )
    y_valid, x_valid = np.where(valid_mask)
    y_missing, x_missing = np.where(~valid_mask)
    
    # Stack into (N, 2) arrays for KDTree
    coords_valid = np.stack([y_valid, x_valid], axis=1)
    coords_missing = np.stack([y_missing, x_missing], axis=1)
    
    # Build Tree on valid pixels
    tree = cKDTree(coords_valid)
    
    # Query nearest neighbor for every missing pixel
    # k=1 returns (distances, indices)
    _, indices = tree.query(coords_missing, k=1)
    
    # Fill values
    # The nearest valid value for each missing pixel
    filled_costmap = costmap.copy()
    filled_costmap[y_missing, x_missing] = costmap[y_valid[indices], x_valid[indices]]
    
    return filled_costmap


def compute_frame_costmap(
    frame_idx: int, 
    pointmap_path: Path
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Worker function to process a single frame.
    1. Projects 3D points to NavMesh (floor-aware).
    2. Computes Geodesic distance.
    3. Fills holes (disconnected islands) via Nearest Neighbor.
    """
    global _worker_navmesh, _worker_goal_snapped, _worker_camera_positions
    
    if _worker_navmesh is None or _worker_goal_snapped is None:
        return frame_idx, None, None

    try:
        if not pointmap_path.exists():
            return frame_idx, None, None
        
        pointmap = np.load(pointmap_path) # (H, W, 3)
        H, W = pointmap.shape[:2]
        
        costmap = np.full((H, W), np.inf, dtype=np.float32)
        valid_mask = np.zeros((H, W), dtype=bool)

        # 1. Identify valid 3D points
        valid_pixels_mask = np.all(np.isfinite(pointmap), axis=2)
        
        if not np.any(valid_pixels_mask):
            return frame_idx, costmap, valid_mask

        rows, cols = np.where(valid_pixels_mask)
        points_3d = pointmap[rows, cols]

        # 2. Compute Geodesic with floor-aware snapping
        goal_pos = _worker_goal_snapped
        nav = _worker_navmesh
        distances = np.full(len(rows), np.inf, dtype=np.float32)
        
        # Get camera position for this frame (if available)
        camera_pos = None
        if _worker_camera_positions is not None and frame_idx in _worker_camera_positions:
            camera_pos = _worker_camera_positions[frame_idx]
        
        path = habitat_sim.ShortestPath()
        path.requested_end = goal_pos

        for i in range(len(rows)):
            pt = points_3d[i]
            # Snap to navigable surface (floor-aware if camera position available)
            if camera_pos is not None:
                snapped_pt = snap_point_to_floor(pt, nav, camera_pos[1])
            else:
                snapped_pt = nav.snap_point(pt)
            
            path.requested_start = snapped_pt
            found = nav.find_path(path)
            
            if found:
                distances[i] = path.geodesic_distance
            # If not found, it remains inf (will be filled later)

        costmap[rows, cols] = distances
        
        # 3. Fill Holes (In-painting)
        # We perform this on the 2D image to fix table-tops/islands
        costmap_filled = fill_holes_nearest(costmap)
        
        # Re-compute valid mask based on the filled map
        # Now valid means "reachable or near something reachable"
        final_valid_mask = np.isfinite(costmap_filled) & valid_pixels_mask

        return frame_idx, costmap_filled, final_valid_mask

    except Exception as e:
        return frame_idx, None, None


def save_visualization(
    costmap: np.ndarray, 
    output_path: Path, 
    cmap: str = 'turbo'
) -> None:
    """Render visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    finite_mask = np.isfinite(costmap)
    
    if not np.any(finite_mask):
        plt.close(fig)
        return

    valid_costs = costmap[finite_mask]
    
    vmin = np.percentile(valid_costs, 1)
    vmax = np.percentile(valid_costs, 99)

    costmap_masked = np.ma.masked_where(~finite_mask, costmap)

    im = ax.imshow(costmap_masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Geodesic Distance (m)', rotation=270, labelpad=20)

    ax.set_title(
        f"Geodesic Costmap (Filled)\n"
        f"Range: [{np.min(valid_costs):.2f}, {np.max(valid_costs):.2f}] m"
    )
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def load_camera_poses(session_folder: Path) -> dict:
    """
    Load camera poses from poses.txt.
    Format: frame x y z qx qy qz qw
    Returns:
        Dictionary mapping image_idx -> camera_position (x, y, z)
    """
    poses_path = session_folder / "poses_odom.txt"
    camera_positions = {}
    
    if poses_path.exists():
        with open(poses_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    frame_idx = int(parts[0])
                    image_idx = frame_idx
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    camera_positions[image_idx] = np.array([x, y, z])
        
        logger.info(f"Loaded {len(camera_positions)} camera poses from poses.txt")
        logger.info("Floor-aware snapping enabled (ceiling → ground floor)")
    else:
        logger.warning(f"poses.txt not found at {poses_path}")
        logger.warning("Using standard snapping (ceiling may snap to upper floors)")
    
    return camera_positions


def resolve_images_dir(session_folder: Path) -> Path:
    for name in (
        "images_fov90",
        "images",
        "images_downsampled_fov120",
        "images_downsampled",
        "rgb",
        "color",
    ):
        candidate = session_folder / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No images directory found in {session_folder}")


def count_frames_from_images(images_dir: Path) -> int:
    count = 0
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        count += len(list(images_dir.glob(ext)))
    return count


def run_session_parallel(
    session_folder: Path,
    goal_frame_idx: int,
    goal_pixel: Tuple[int, int],
    navmesh_path: Path,
    output_dir: Optional[Path] = None,
    num_workers: int = 8,
    frame_range: Optional[Tuple[int, int]] = None
) -> None:
    
    # 1. Setup paths
    images_dir = resolve_images_dir(session_folder)
    num_frames = count_frames_from_images(images_dir)
    if num_frames <= 0:
        raise ValueError(f"No images found in {images_dir}")

    if frame_range is None:
        start_f, end_f = 0, num_frames - 1
    else:
        start_f, end_f = frame_range
    
    pointmap_dir = session_folder / "gt_pointmaps_fov90"
    if not pointmap_dir.exists() or not any(pointmap_dir.glob("*.npy")):
        print("gt_pointmaps_fov90 doesn't exist!")
        pointmap_dir = session_folder / "pointmaps"
    if not pointmap_dir.exists() or not any(pointmap_dir.glob("*.npy")):
        raise FileNotFoundError(
            f"No pointmaps found in {session_folder / 'gt_pointmaps_fov90'} or "
            f"{session_folder / 'pointmaps'}"
        )
    
    if output_dir is None:
        output_dir = session_folder / f"costmaps_goal{goal_frame_idx:05d}"
    
    arrays_dir = output_dir / "arrays"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    # 2. Main Thread Setup
    logger.info("Initializing Main Process & Goal Snap for frames %d to %d...", start_f, end_f)
    
    sim_cfg = habitat_sim.SimulatorConfiguration()
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    
    if not sim.pathfinder.load_nav_mesh(str(navmesh_path)):
        raise RuntimeError("Failed to load navmesh in main thread")
    
    navmesh = sim.pathfinder
    
    # Load goal
    goal_pmap_path = pointmap_dir / f"{goal_frame_idx:05d}.npy"
    if not goal_pmap_path.exists():
        raise FileNotFoundError(f"Goal pointmap missing: {goal_pmap_path}")
    
    goal_pmap = np.load(goal_pmap_path)
    u, v = goal_pixel
    goal_raw = goal_pmap[v, u]
    
    if not np.all(np.isfinite(goal_raw)):
        raise ValueError(f"Goal point {goal_raw} is invalid/infinite.")
        
    goal_snapped = navmesh.snap_point(goal_raw)
    logger.info(f"Goal: {goal_raw} -> Snapped: {goal_snapped}")
    
    # Load camera poses for floor-aware snapping
    camera_positions = load_camera_poses(session_folder)
    
    # 3. Parallel Execution
    logger.info(f"Starting pool with {num_workers} workers...")
    
    tasks = []
    
    with ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=init_worker, 
        initargs=(str(navmesh_path), goal_snapped, camera_positions)
    ) as executor:
        
        for i in range(start_f, end_f + 1):
            p_path = pointmap_dir / f"{i:05d}.npy"
            if p_path.exists():
                tasks.append(executor.submit(compute_frame_costmap, i, p_path))

        success = 0
        failed = 0
        
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Computing Costmaps"):
            f_idx, costmap, valid_mask = future.result()
            
            if costmap is None:
                failed += 1
                continue
            
            np.save(arrays_dir / f"{f_idx:05d}.npy", costmap)
            save_visualization(costmap, output_dir / f"{f_idx:05d}.png")
            success += 1

    logger.info(f"Done. Success: {success}, Failed/Skipped: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute geodesic costmaps (Parallel + Hole Filling).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--session_folder", type=Path, required=True)
    parser.add_argument("--goal_frame", type=int, required=True)
    parser.add_argument("--goal_pixel_u", type=int, required=True)
    parser.add_argument("--goal_pixel_v", type=int, required=True)
    parser.add_argument("--navmesh_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processes")
    args = parser.parse_args()

    run_session_parallel(
        session_folder=args.session_folder,
        goal_frame_idx=args.goal_frame,
        goal_pixel=(args.goal_pixel_u, args.goal_pixel_v),
        navmesh_path=args.navmesh_path,
        output_dir=args.output_dir,
        num_workers=args.workers
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()