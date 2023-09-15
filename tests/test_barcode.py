import numpy as np
import pytest

from pysynch.io import DigitalIntanData


@pytest.fixture(scope="function")
def intan_data(intan_data_path):
    return DigitalIntanData.from_folder(
        intan_data_path, dig_channel_names=["barcodes", "camera"]
    )


def test_barcode(intan_data):
    barcode_tsd = intan_data.barcodes_tsd

    assert len(barcode_tsd.barcodes) == 14
    print([barcode_tsd.barcodes[i] for i in [0, -1]])
    print([f(barcode_tsd.index) for f in [min, max]])
    assert np.allclose(
        [barcode_tsd.barcodes[i] for i in [0, -1]], [4294940530, 4294940543]
    )
    assert np.allclose([f(barcode_tsd.index) for f in [min, max]], [0, 70.861])
