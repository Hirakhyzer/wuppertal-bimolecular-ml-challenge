from pathlib import Path

import numpy as np
import pytest

from src.xyz_parser import atom_count_summary, read_concatenated_xyz


def test_parses_two_concatenated_frames(tmp_path: Path):
    xyz = tmp_path / "toy.xyz"
    xyz.write_text(
        "2\nfirst\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7\n\n"
        "1\nsecond\nHe 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    frames = read_concatenated_xyz(xyz)
    assert len(frames) == 2
    assert frames[0].atom_count == 2
    assert frames[1].symbols == ("He",)
    assert np.allclose(frames[0].coordinates[1], [0.0, 0.0, 0.7])
    assert atom_count_summary(frames)["distribution"] == {"1": 1, "2": 1}


def test_rejects_truncated_xyz_frame(tmp_path: Path):
    xyz = tmp_path / "broken.xyz"
    xyz.write_text("2\ncomment\nH 0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ended early"):
        read_concatenated_xyz(xyz)
