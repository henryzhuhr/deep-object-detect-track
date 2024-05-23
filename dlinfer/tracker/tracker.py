import copy

import numpy as np

from .byte_tracker import ByteTracker


class Tracker:
    backends = [ByteTracker]

    def __init__(self):
        self.tracker = ByteTracker(frame_rate=30)
