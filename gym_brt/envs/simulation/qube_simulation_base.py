from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


class QubeSimulatorBase(object):
    """Base class for the simulators that has the same interface as the hardware wrapper."""

    def __init__(self, frequency=250, integration_steps=1, max_voltage=18.0):
        self._dt = 1.0 / frequency
        self._integration_steps = integration_steps
        self._max_voltage = max_voltage
        self.state = (
            np.array([0, 0, 0, 0], dtype=np.float64) #+ np.random.randn(4) * 0.01
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def step(self, action, led=None):
        raise NotImplementedError

    def reset_up(self):
        raise NotImplementedError

    def reset_down(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def reset_encoders(self):
        pass

    def close(self, type=None, value=None, traceback=None):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}