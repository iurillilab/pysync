import numpy as np
import pytest

from pysynch import DigitalTsd


@pytest.mark.parametrize(
    "time_kwargs", [dict(), dict(time_array=np.arange(10), rate=2.0)]
)
def test_notime_error(time_kwargs):
    with pytest.raises(AssertionError):
        DigitalTsd([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], **time_kwargs)


@pytest.mark.parametrize(
    "time_kwargs", [dict(time_array=np.arange(0, 5, 0.5)), dict(rate=2.0)]
)
class TestDigitalTsd:
    def test_properties(self, time_kwargs):
        signal = DigitalTsd([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], **time_kwargs)
        assert np.allclose(signal.onset_idxs, [1, 6])
        assert np.allclose(signal.offset_idxs, [4, 8])
        assert np.allclose(signal.all_event_idxs, [1, 4, 6, 8])

    def test_event_times(self, time_kwargs):
        signal = DigitalTsd([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], **time_kwargs)
        assert np.allclose(signal.onset_times.index, [0.5, 3.0])
        assert np.allclose(signal.offset_times.index, [2.0, 4.0])
        assert np.allclose(signal.all_event_times.index, [0.5, 2.0, 3.0, 4.0])
