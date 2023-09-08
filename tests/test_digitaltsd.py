from audioop import ratecv
from mailbox import _SupportsReadAndReadline
from operator import ne
from pstats import StatsProfile
from runpy import run_module
from time import localtime
from igraph import ClusterColoringPalette
from jupyter_server import DEFAULT_STATIC_FILES_PATH
from napari import view_tracks
from networkx import fast_could_be_isomorphic, generate_multiline_adjlist
import numpy as np
from psutil import pid_exists
from sklearn.dummy import check_random_state
from sklearn.inspection import PartialDependenceDisplay
from uri_template import ExpansionReservedError
from pysynch import DigitalTsd
import pytest

from symbol import annassign


@pytest.mark.parametrize("time_kwargs", [dict(), dict(time_array=np.arange(10), rate=2.0)])
def test_notime_error(time_kwargs):
    with pytest.raises(AssertionError):
        DigitalTsd([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], **time_kwargs)


@pytest.mark.parametrize(
    "array_kwargs, time_kwargs", [[dict(), dict()], [dict(time_array=np.arange(0, 5, 0.5)), dict(rate=2.0)]]
)
class TestDigitalTsd:
    def test_properties(self, time_kwargs):
        signal = DigitalTsd([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], **time_kwargs)
        assert np.allclose(signal.onsets, [1, 6])
        assert np.allclose(signal.offsets, [4, 8])
        assert np.allclose(signal.all_events, [1, 4, 6, 8])
    
    def test_event_times(self, time_kwargs):
        signal = DigitalTsd([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], **time_kwargs)
        assert np.allclose(signal.onsets_times.index, [0.5, 3.0])
        assert np.allclose(signal.offsets_times.index, [2.0, 4.0])
        assert np.allclose(signal.all_events_times.index, [0.5, 2.0, 3.0, 4.0])
