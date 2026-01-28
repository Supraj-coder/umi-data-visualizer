import numpy as np
import zarr
import pytest
from pathlib import Path
from main import UMIFrame, VisConfig, run_visualizer, AppSettings

def test_umiframe_validation():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pos = np.array([1.0, 2.0, 3.0])
    rot = np.array([0.1, 0.2, 0.3])
    
    frame = UMIFrame(image=img, eef_pos=pos, eef_rot=rot, gripper_width=0.5, index=0)
    assert frame.index == 0

    with pytest.raises(ValueError, match="eef_pos must be shape"):
        UMIFrame(image=img, eef_pos=np.array([1.0, 2.0]), eef_rot=rot, gripper_width=0.5, index=0)

    with pytest.raises(ValueError, match="Image data is empty"):
        UMIFrame(image=np.array([]), eef_pos=pos, eef_rot=rot, gripper_width=0.5, index=0)

@pytest.fixture
def mock_zarr_structure(tmp_path):
    parent_dir = tmp_path / "project_dir"
    parent_dir.mkdir()
    
    zarr_path = parent_dir / "dataset.zarr"
    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store)
    
    meta = root.create_group("meta")
    meta["episode_ends"] = np.array([2], dtype=np.int64)
    
    data = root.create_group("data")
    data["camera0_rgb"] = np.zeros((2, 10, 10, 3), dtype=np.uint8)
    data["robot0_eef_pos"] = np.random.rand(2, 3).astype(np.float64)
    data["robot0_eef_rot_axis_angle"] = np.random.rand(2, 3).astype(np.float64)
    data["robot0_gripper_width"] = np.zeros((2, 1), dtype=np.float64)
    
    return parent_dir

def test_integration_run(mock_zarr_structure):
    config = VisConfig(
        zarr_path=mock_zarr_structure,
        episode_id=0,
        settings=AppSettings()
    )
    
    run_visualizer(config)

def test_vis_config_invalid_path():
    with pytest.raises(FileNotFoundError):
        VisConfig(zarr_path=Path("/tmp/non_existent_path_12345"), episode_id=0)

def test_vis_config_invalid_episode(tmp_path):
    dummy_path = tmp_path / "dummy.zarr"
    dummy_path.mkdir()
    with pytest.raises(ValueError, match="episode_id must be >= 0"):
        VisConfig(zarr_path=dummy_path, episode_id=-1)
