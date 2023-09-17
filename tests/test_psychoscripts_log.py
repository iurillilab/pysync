from pathlib import Path

import numpy as np
import pytest

from pysynch.io.psychoscripts_log import (
    PsychoScriptsLog,
    PsychoScriptsLogFolder,
)


@pytest.fixture(scope="session")
def psychoscripts_data_path():
    # TODO in the future deal with future versions
    return Path(__file__).parent / "assets" / "psychoscripts_data" / "20230901"


@pytest.mark.parametrize(
    "filename, results",
    [
        (
            "*_servo_visual_looming_M9.log",
            dict(
                all_lines_len=2614,
                barcode_lines_len=221,
                cpu_timebase_len=172,
                cpu_timebase_min_max=[1.6935609110233362e18, 1.6935659386660385e18],
                psy_timebase_min_max=[5.222, 5032.6876],
            ),
        ),
        (
            "*visual_moving_2d_target_M9.log",
            dict(
                all_lines_len=345,
                barcode_lines_len=81,
                cpu_timebase_len=6,
                cpu_timebase_min_max=[1.6935670399618355e18, 1.6935673849721687e18],
                psy_timebase_min_max=[4.1699, 349.1758],
            ),
        ),
    ],
)
def test_file_loading(psychoscripts_data_path, filename, results):
    data = next(psychoscripts_data_path.glob(filename))

    log = PsychoScriptsLog(data)

    assert len(log.all_lines) == results["all_lines_len"]

    assert len(log.barcode_lines) == results["barcode_lines_len"]

    assert len(log.cpu_timebase.index) == results["cpu_timebase_len"]
    assert len(log.psy_timebase.index) == len(log.cpu_timebase.index)
    assert np.allclose(
        np.array([log.cpu_timebase.index.min(), log.cpu_timebase.index.max()])
        - np.array(results["cpu_timebase_min_max"]),
        [0, 0],
    )
    assert np.allclose(
        [log.psy_timebase.index.min(), log.psy_timebase.index.max()],
        results["psy_timebase_min_max"],
    )
    print(log.cpu_timebase.index.min(), log.cpu_timebase.index.max())


def test_psychoscripts_log_folder(psychoscripts_data_path):
    log_folder = PsychoScriptsLogFolder(psychoscripts_data_path)
    df = log_folder.stim_epochs_df

    # TODO this could be tested more
    assert np.allclose(df.isvisual, [True, True, False, False])
    assert np.allclose(df.repetition, [0, 0, 0, 1])
    assert len(log_folder.stim_logs_list) == 4
    assert len(log_folder.file_list) == 4
