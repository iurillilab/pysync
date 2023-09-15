from pathlib import Path
from re import I
import pytest
import numpy as np
from pysynch.core.digital_signal import DigitalTsd
from pysynch.io.intan_data import DigitalIntanData

# from conftest import intan_data_path

COLNAMES = ["barcodes", "camera"]

@pytest.fixture(scope="function")
def intan_data(intan_data_path):
    return DigitalIntanData.from_folder(intan_data_path, dig_channel_names=COLNAMES)


def test_intan_data_caching(intan_data_path):
    def _find_npz_files(path):
        return list(path.glob(DigitalIntanData.CACHED_FILE_TEMPLATE_NAME.format("*")))

    # Check that there's no npz file in the folder before loading:
    assert len(_find_npz_files(intan_data_path)) == 0
    intan_data = DigitalIntanData.from_folder(intan_data_path)

    # Check that there's a npz file in the folder after loading and try loading it:
    assert len(_find_npz_files(intan_data_path)) == 1

    # Channel names should be disregarded when loading from cache:
    
    intan_data = DigitalIntanData.from_folder(intan_data_path, dig_channel_names=COLNAMES)
    assert all([name == col for name, col in zip(intan_data.columns, [0, 1])])

    # test forcing reload:
    intan_data = DigitalIntanData.from_folder(
        intan_data_path, dig_channel_names=COLNAMES, force_loading=True
    )
    assert all([name == col for name, col in zip(intan_data.columns, COLNAMES)])


@pytest.mark.parametrize(
    "kwargs, columns",
    [
        (dict(), [0, 1]),
        (dict(dig_channel_names=COLNAMES), COLNAMES),
    ],
)
def test_intan_data_load(intan_data_path, kwargs, columns):
    intan_data = DigitalIntanData.from_folder(intan_data_path, **kwargs)
    assert all([name == col for name, col in zip(intan_data.columns, columns)])
    assert len(intan_data) == 70860
    assert intan_data.shape == (70860, 2)

    # test caching of cols names:
    intan_data = DigitalIntanData.from_folder(intan_data_path) 

    assert all([name == col for name, col in zip(intan_data.columns, columns)])
 

def test_intan_data_slice(intan_data):
    assert isinstance(intan_data, DigitalIntanData)
    assert isinstance(intan_data[COLNAMES], DigitalIntanData)
    assert isinstance(intan_data[COLNAMES[0]], DigitalTsd)


