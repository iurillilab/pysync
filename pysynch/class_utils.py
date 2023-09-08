def lazy_property(func):
    attribute_name = "_" + func.__name__

    @property
    def wrapper(self):
        if not hasattr(self, attribute_name):
            setattr(self, attribute_name, func(self))
        return getattr(self, attribute_name)

    return wrapper
