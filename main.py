import rerun as rr
import zarr
import numpy as np
import argparse
import itertools
import sys
from pathlib import Path
from typing import List, Tuple
from imagecodecs.numcodecs import register_codecs
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

register_codecs()

@dataclass
class VisConfig:
    zarr_path: Path
    episode_id: int
    trajectory_color: Tuple[int,int,int] = (100,149,237)
    gripper_color: Tuple[int,int,int] = (255,165,0)

@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class UMIFrame:
    image: np.ndarray
    eef_pos: np.ndarray
    eef_rot: np.ndarray
    gripper_width: float
    index: int

def stream_to_rerun(frame: UMIFrame, config: VisConfig, trajectory: List[np.ndarray]) -> None:
    rr.set_time_sequence(timeline="frame_ids", sequence=frame.index)
    
    rot_vec = frame.eef_rot
    angle = float(np.linalg.norm(rot_vec))
    rotation = rr.RotationAxisAngle(axis=rot_vec/angle, angle=angle) if angle > 1e-6 else None
    rr.log("world/robot/ee_pos", rr.Transform3D(translation=frame.eef_pos, rotation=rotation))
    
    rr.log(
        "world/robot/ee_pos/camera",
        rr.Pinhole(
            focal_length=450, 
            width=frame.image.shape[1], 
            height=frame.image.shape[0],
            image_plane_distance=0.2
        )
    )
    rr.log("world/robot/ee_pos/camera", rr.Image(frame.image))

    trajectory.append(frame.eef_pos)
    rr.log("world/robot/trajectory", rr.LineStrips3D([trajectory], colors=config.trajectory_color, radii=0.002))
    rr.log(
        "world/robot/ee_pos/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            radii=0.002
        )
    )
    rr.log("world/robot/gripper_width", rr.Scalar(frame.gripper_width))

def run_visualizer(config: VisConfig) -> None:
	if not config.zarr_path.exists():
		print("Path does not exists.")
		sys.exit(1)
	try:
		with zarr.open(str(config.zarr_path), 'r') as root:
			print("Number of episodes in the dataset(s):",len(root["meta/episode_ends"]))
			ends = root["meta/episode_ends"][:]
			start_id = 0 if config.episode_id == 0 else int(ends[config.episode_id - 1])
			end_id = int(ends[config.episode_id])
			rr.init("UMI_Full_Visualizer", spawn=True)
			rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
			img_stream = itertools.islice(root["data/camera0_rgb"], start_id, end_id)
			pos_arr = root["data/robot0_eef_pos"][start_id:end_id]
			rot_arr = root["data/robot0_eef_rot_axis_angle"][start_id:end_id]
			grp_arr = root["data/robot0_gripper_width"][start_id:end_id]
			trajectory: List[np.ndarray] = []
			for i, img in enumerate(img_stream):
				if img is None or img.size == 0:
					print(f"Skipping frame {i}: Image data is empty.")
					continue
				try:
					frame = UMIFrame(img, pos_arr[start_id+i], rot_arr[start_id+i], float(grp_arr[start_id+i][0]), i)
				except Exception as e:
					print(f"Error processing frame {i}: {e}")
					break
				stream_to_rerun(frame, config, trajectory)
	except Exception as e:
		print(e)
		sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=lambda p: Path(p).expanduser().resolve())
    parser.add_argument("--episode", type=int, default=1)
    args = parser.parse_args()
    run_visualizer(VisConfig(zarr_path=args.path, episode_id=args.episode-1))

if __name__ == "__main__":
    main()
