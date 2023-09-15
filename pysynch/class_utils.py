from functools import wraps
from pathlib import Path

import numpy as np
from typing_extensions import deprecated


@deprecated("Use the itertools.cached_property decorator instead")
def attribute_caching_property(func: callable):
    """Property to cache the results of a method in a private attribute of the same name.

    Parameters
    ----------
    func : method
        A method to decorate

    Returns
    -------
    callable
        The decorated method
    """

    attribute_name = "_" + func.__name__

    @property
    @wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute_name):
            setattr(self, attribute_name, func(self))
        return getattr(self, attribute_name)

    return wrapper


FILE_CACHE_ATTRIBUTE_NAME = "_file_cache_path"
CACHE_SUFFIX = "_cached.npy"


def file_caching_property(func: callable):
    """_summary_

    Parameters
    ----------
    func : callable
       The function to decorate

    Returns
    -------
    callable
       The decorated function
    """

    filename = func.__name__ + CACHE_SUFFIX

    @property
    @wraps(func)
    def wrapper(self):
        assert hasattr(
            self, FILE_CACHE_ATTRIBUTE_NAME
        ), "The object must have a file_cache_path attribute!"
        data_path = Path(getattr(self, FILE_CACHE_ATTRIBUTE_NAME))
        data_path.mkdir(exist_ok=True, parents=True)

        if (data_path / filename).exists():
            return np.load(data_path / filename)
        else:
            result = func(self)
            np.save(data_path / filename, result)
            return result

    return wrapper
