import pytest


# Define a fixture to create a temporary folder
@pytest.fixture(scope="session")
def temporary_cache_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cache")
    return tmp_path

