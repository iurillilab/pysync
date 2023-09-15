from pathlib import Path

from pysynch.class_utils import (
    FILE_CACHE_ATTRIBUTE_NAME,
    file_caching_property,
)


class DemoClass:
    def __init__(self, a, b, filecache_path: str | Path = None):
        self.a = a
        self.b = b
        setattr(self, FILE_CACHE_ATTRIBUTE_NAME, filecache_path)

    @file_caching_property
    def file_cached(self):
        return self.a - self.b


# TODO test when implemented
# def test_file_caching(temporary_cache_dir):
#     cache_dir = temporary_cache_dir / "cache"
#     an_obj = DemoClass(1, 2, filecache_path=cache_dir)
#     assert not cache_dir.exists()

#     an_obj.file_cached

#     assert cache_dir.exists()
#     assert (cache_dir / f"file_cached{CACHE_SUFFIX}").exists()
