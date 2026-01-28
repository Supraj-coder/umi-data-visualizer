import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rerun as rr
import zarr
from imagecodecs.numcodecs import register_codecs
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass
class AppSettings:
    focal_length: float = 450.0
    image_plane_distance: float = 0.2
    trajectory_color: tuple[int, int, int] = (100, 149, 237)
    gripper_color: tuple[int, int, int] = (255, 165, 0)
    trajectory_radii: float = 0.002


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VisConfig:
    zarr_path: Path
    episode_id: int
    settings: AppSettings = AppSettings()

    def __post_init__(self) -> None:
        self.zarr_path = self.zarr_path.expanduser().resolve()
        if self.episode_id < 0:
            raise ValueError(f"episode_id must be >= 0, got {self.episode_id}")
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Path not found: {self.zarr_path}")


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class UMIFrame:
    image: npt.NDArray[np.uint8]
    eef_pos: npt.NDArray[np.float64]
    eef_rot: npt.NDArray[np.float64]
    gripper_width: float
    index: int

    def __post_init__(self) -> None:
        if self.eef_pos.shape != (3,):
            raise ValueError(
                f"Frame {self.index}: eef_pos must be shape (3,), got {self.eef_pos.shape}"
            )

        if self.eef_rot.shape != (3,):
            raise ValueError(
                f"Frame {self.index}: eef_rot must be shape (3,), got {self.eef_rot.shape}"
            )

        if self.image.size == 0:
            raise ValueError(f"Frame {self.index}: Image data is empty.")


def stream_to_rerun(
    frame: UMIFrame, config: VisConfig, trajectory: list[npt.NDArray[np.float64]]
) -> None:
    rr.set_time_sequence(timeline="frame_ids", sequence=frame.index)

    rot_vec = frame.eef_rot
    angle = float(np.linalg.norm(rot_vec))
    rotation = (
        rr.RotationAxisAngle(axis=rot_vec / angle, angle=angle)
        if angle > 1e-6
        else None
    )
    rr.log(
        "world/robot/ee_pos",
        rr.Transform3D(translation=frame.eef_pos, rotation=rotation),
    )

    rr.log(
        "world/robot/ee_pos/camera",
        rr.Pinhole(
            focal_length=config.settings.focal_length,
            width=frame.image.shape[1],
            height=frame.image.shape[0],
            image_plane_distance=config.settings.image_plane_distance,
        ),
    )
    rr.log("world/robot/ee_pos/camera", rr.Image(frame.image))

    trajectory.append(frame.eef_pos)
    rr.log(
        "world/robot/trajectory",
        rr.LineStrips3D(
            [trajectory],
            colors=config.settings.trajectory_color,
            radii=config.settings.trajectory_radii,
        ),
    )
    rr.log(
        "world/robot/ee_pos/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            radii=0.002,
        ),
    )
    rr.log("world/robot/gripper_width", rr.Scalar(frame.gripper_width))


def run_visualizer(config: VisConfig) -> None:
    if (not config.zarr_path.exists()) or (
        not (config.zarr_path / "dataset.zarr").exists()
    ):
        print("Path does not exists.")
        sys.exit(1)
    if (config.zarr_path / "dataset.zarr").exists():
        config.zarr_path = config.zarr_path / "dataset.zarr"
    with zarr.open(str(config.zarr_path), "r") as root:
        print("Number of episodes in the dataset(s):", len(root["meta/episode_ends"]))
        ends = root["meta/episode_ends"][:]
        start_id = 0 if config.episode_id == 0 else int(ends[config.episode_id - 1])
        end_id = int(ends[config.episode_id])
        rr.init("UMI_Full_Visualizer", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        img_stream = itertools.islice(root["data/camera0_rgb"], start_id, end_id)
        pos_arr = root["data/robot0_eef_pos"][start_id:end_id]
        rot_arr = root["data/robot0_eef_rot_axis_angle"][start_id:end_id]
        grp_arr = root["data/robot0_gripper_width"][start_id:end_id]
        trajectory: list[npt.NDArray[np.float64]] = []
        for i, img in enumerate(img_stream):
            if img is None or img.size == 0:
                print(f"Skipping frame {i}: Image data is empty.")
                continue
            frame = UMIFrame(img, pos_arr[i], rot_arr[i], float(grp_arr[i][0]), i)
            stream_to_rerun(frame, config, trajectory)


def main() -> None:

    register_codecs()
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=lambda p: Path(p).expanduser().resolve())
    parser.add_argument("--episode", type=int, default=1)
    args = parser.parse_args()
    run_visualizer(VisConfig(zarr_path=args.path, episode_id=args.episode - 1))


if __name__ == "__main__":
    main()
