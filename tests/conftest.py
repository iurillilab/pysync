import shutil
from pathlib import Path

import pytest

# Define a fixture to create a temporary folder
# @pytest.fixture(scope="session")
# def temporary_cache_dir(tmp_path_factory):


@pytest.fixture(scope="function")
def intan_data_path(tmp_path_factory):
    SOURCE_INTAN_FOLDER = Path(__file__).parent / "assets" / "intan_data"

    intan_data_path = Path(tmp_path_factory.mktemp("intan_data"))
    # intan_data_path = tmp_path / "intan_data"
    for file in SOURCE_INTAN_FOLDER.iterdir():
        print(file)
        shutil.copy(file, intan_data_path / file.name)
    return intan_data_path
