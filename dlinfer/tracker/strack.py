from __future__ import annotations

import numpy as np

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .basetrack import BaseTrack, TrackState


from collections import deque


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    @property
    def tlwh(self):
        """current bounding box format `(top left x, top left y, width, height)`."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,`(top left, bottom right)`."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def __init__(self, tlwh: np.ndarray, score: np.float64, clsid: int, history_len=1):
        # wait activate
        # tlwh: box format (x1,y1,w,h)/(top left x, top left y, width, height)
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.clsid = clsid

        self.__history_track = deque(maxlen=history_len)
        self.__history_track.append(self._tlwh)
        print(f"self.__history_track: {self.__history_track}")

        self.kalman_filter: KalmanFilter = None
        self.mean: np.ndarray = None
        self.covariance: np.ndarray = None
        self.is_activated: bool = False

        self.tracklet_len: int = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(
        self,
        new_track: STrack,  # `from __future__ import annotations` , https://peps.python.org/pep-0563/
        frame_id: int,
    ):
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
