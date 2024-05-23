from typing import List
import numpy as np

from . import matching

from .kalman_filter import KalmanFilter
from .basetrack import TrackState

from .strack import STrack


class ByteTracker:
    def __init__(
        self,
        frame_rate: int = 30,
        track_thresh: float = 0.4,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.track_thresh = track_thresh
        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.match_thresh = match_thresh
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(
        self,
        dets: np.ndarray,  # [B, 6]: [x1, y1, x2, y2, conf, cls]
        scale_h=1.0,
        scale_w=1.0,
    ):
        self.frame_id += 1
        activated_starcks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        # [x1, y1, x2, y2, conf, cls]
        # [ 0,  1,  2,  3,   4,   5]
        bboxes = dets[:, :4].astype(np.int32)
        scores = dets[:, 4]
        clsids = dets[:, 5].astype(np.int32)

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)

        # First association
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        clsids = clsids[remain_inds]
        # Second association
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        clsids_second = clsids[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = []
            for i_det in range(dets.shape[0]):
                bbox = dets[i_det]
                score = scores_keep[i_det]
                clsid = clsids[i_det]
                x1 = int(scale_w * bbox[0])
                y1 = int(scale_h * bbox[1])
                x2 = int(scale_w * bbox[2])
                y2 = int(scale_h * bbox[3])
                tlwh = np.array([x1, y1, w := x2 - x1, h := y2 - y1])
                detections.append(STrack(tlwh, score, clsid))
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        multi_predict(strack_pool)

        dists = matching.iou_distance(strack_pool, detections)
        # if not self.args.mot20: # TODO
        #     dists = matching.fuse_score(dists, detections)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            itracked: int
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = []
            for i_det in range(dets_second.shape[0]):
                bbox = dets_second[i_det]
                score = scores_second[i_det]
                clsid = clsids_second[i_det]
                x1 = int(scale_w * bbox[0])
                y1 = int(scale_h * bbox[1])
                x2 = int(scale_w * bbox[2])
                y2 = int(scale_h * bbox[3])
                tlwh = np.array([x1, y1, w := x2 - x1, h := y2 - y1])
                detections_second.append(STrack(tlwh, score, clsid))
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        output_stracks: List[STrack]

        return output_stracks


def multi_predict(stracks: List[STrack]):
    if len(stracks) > 0:
        lmm = []
        for st in stracks:
            lmm.append(st.mean.copy())
        multi_mean = np.asarray(lmm)
        lmc = []
        for st in stracks:
            lmc.append(st.covariance)
        multi_covariance = np.asarray(lmc)

        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]):
    exists = {}
    res: List[STrack] = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
