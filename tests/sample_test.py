from pynapple.core import Tsd, TsdFrame
import numpy as np



class TsdSubclass(Tsd):
    def copy(self) -> np.ndarray:
        return self.__class__(d=self.values.copy(), t=self.index.copy())
    pass

# a = TsdSubclass(d=np.array([1,2,3, 2, 1]), t=np.array([0, 0.5, 1, 1.5, 2]))
a = Tsd(d=np.array([True, False, True, True, False]), t=np.array([0, 0.5, 1, 1.5, 2]))



b = TsdSubclass(d=np.array([1,2,3, 2, 1]), t=np.array([0, 0.5, 1, 1.5, 2])+0.5)
print(b.values)
#np.diff(a)
a_copy = b.copy()
np.insert(a.values[:-1] & a.values[1:], 0, False)
# np.concatenate([a, a])