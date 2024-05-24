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
        bboxes = dets[:, :4]
        scores = dets[:, 4]
        clsids = dets[:, 5].astype(np.int32)

        # 找到置信度高的框，作为第一次关联的框
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        # 找到置信度低的框，作为第二次关联的框
        # First association
        inds_second = np.logical_and(inds_low, inds_high)
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
                [x1, y1, x2, y2] = [scale_w * bbox[0], scale_h * bbox[1], scale_w * bbox[2], scale_h * bbox[3]]
                tlwh = np.array([x1, y1, w := x2 - x1, h := y2 - y1])
                detections.append(STrack(tlwh, score, clsid))
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:  # 第一次出现的目标，未激活跟踪
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        # 将已经追踪到的track和丢失的track合并
        # 其中注意：丢失的 track 是指某一帧可能丢了一次，但是仍然在缓冲帧范围之内，所以依然可以用来匹配
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        multi_predict(strack_pool)  # 先用卡尔曼滤波预测每一条轨迹在当前帧的位置

        # 让预测后的track和当前帧的detection框做cost_matrix，用的方式为 IOU 关联
        # 这里的iou_distance 函数中调用了track.tlbr，返回的是预测之后的 track 坐标信息
        dists = matching.iou_distance(strack_pool, detections)
        # if not self.args.mot20: # TODO
        #     dists = matching.fuse_score(dists, detections)

        # 用匈牙利算法算出相匹配的 track 和 detection 的索引，
        # 以及没有被匹配到的 track 和没有被匹配到的 detection 框的索引
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            itracked: int
            # 找到匹配到的所有track&detection pair 并且用 detection 来更新卡尔曼的状态
            track = strack_pool[itracked]
            det = detections[idet]
            # 对应 strack_pool 中的 tracked_stracks
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # 对应 strack_pool 中的 self.lost_stracks，重新激活 track
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
                [x1, y1, x2, y2] = [scale_w * bbox[0], scale_h * bbox[1], scale_w * bbox[2], scale_h * bbox[3]]
                tlwh = np.array([x1, y1, w := x2 - x1, h := y2 - y1])
                detections_second.append(STrack(tlwh, score, clsid))
        else:
            detections_second = []

        # 找出strack_pool中没有被匹配到的track（这帧目标被遮挡的情况）
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # 在低置信度的检测框中再次与没有被匹配到的 track 做 IOU 匹配
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 如果 track 经过两次匹配之后还没有匹配到 box 的话，就标记为丢失了
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        # 处理第一次匹配时没有被track匹配的检测框（一般是这个检测框第一次出现的情形）
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)  # 计算未被匹配的框和不确定的track之间的cost_matrix
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        # 匹配不上的unconfirmed_track直接删除，说明这个track只出现了一帧
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 经过上面这些步骤后，如果还有没有被匹配的检测框，说明可能画面中新来了一个物体，
        # 那么就直接将他视为一个新的track，但是这个track的状态并不是激活态。
        # 在下一次循环的时候会先将它放到unconfirmed_track中去，
        # 然后根据有没有框匹配它来决定是激活还是丢掉
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # 对于丢失目标的track来说，判断它丢失的帧数是不是超过了buffer缓冲帧数，超过就删除
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 指上一帧匹配上的track
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 加上这一帧新激活的track（两次匹配到的track，以及由unconfirm状态变为激活态的track
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 加上丢帧目标重新被匹配的track
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks 在经过这一帧的匹配之后如果被重新激活的话就将其移出列表
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 将这一帧丢掉的track添加进列表
        self.lost_stracks.extend(lost_stracks)
        # self.lost_stracks 如果在缓冲帧数内一直没有被匹配上被 remove 的话也将其移出 lost_stracks 列表
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 更新被移除的track列表
        self.removed_stracks.extend(removed_stracks)
        # 将这两段 track 中重合度高的部分给移除掉
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        # 得到最终的结果，也就是成功追踪的track序列
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


def sub_stracks(tlista: List[STrack], tlistb: List[STrack]):
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
