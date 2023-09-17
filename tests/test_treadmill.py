from pathlib import Path

import pytest

from pysynch.io.treadmill_data import TreadmillData


@pytest.fixture(scope="session")
def treadmill_data_path():
    return Path(__file__).parent / "assets" / "treadmill_data"


def test_path_loading_error():
    # test exception when no files are found:
    with pytest.raises(AssertionError):
        TreadmillData.from_folder("no_files")


def test_path_loading(treadmill_data_path):
    treadmill_data = TreadmillData.from_folder(treadmill_data_path)

    assert len(treadmill_data) == 974
    assert treadmill_data.index.min() == 0.0
    assert treadmill_data.index.max() == 3046.984089088
    assert all(
        [
            a == b
            for a, b in zip(
                treadmill_data.columns,
                ["t_ns", "pitch", "roll", "yaw", "x0", "y0", "x1", "y1"],
            )
        ]
    )


def test_file_loading(treadmill_data_path):
    treadmill_data = TreadmillData.from_csv(
        sorted(treadmill_data_path.glob("*_data.csv"))[0]
    )

    assert len(treadmill_data) == 162
    assert treadmill_data.index.min() == 0.0
    assert treadmill_data.index.max() == 0.4135844
    assert all(
        [
            a == b
            for a, b in zip(
                treadmill_data.columns,
                ["t_ns", "pitch", "roll", "yaw", "x0", "y0", "x1", "y1"],
            )
        ]
    )
