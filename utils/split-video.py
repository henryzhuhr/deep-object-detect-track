import os
import copy
import time
import argparse

import cv2
from matplotlib.pylab import f


class TrackArgs:

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        # fmt: off
        parser.add_argument("-v", "--video", type=str)    
        parser.add_argument("-o", "--outdir", type=str, default=None)
        parser.add_argument("-r", "--spilt-fps", type=int, default=10)
        parser.add_argument("-s", "--max-size", type=int, default=1080)
        # fmt: on

        return parser.parse_args()

    def __init__(self) -> None:
        args = self.get_args()
        # fmt: off
        self.video: str = os.path.expandvars(os.path.expanduser(args.video))
        if not os.path.exists(self.video):
            raise FileNotFoundError(f"Video file not found: {self.video}")
        # fmt: on
        if self.video[-1] in ["/", "\\"]:
            self.video = self.video[:-1]

        self.outdir: str = args.outdir
        if self.outdir is None:
            self.outdir = os.path.splitext(self.video)[0]

        self.spilt_fps: int = args.spilt_fps


def main():
    args = TrackArgs()
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        print(f"Directory '{outdir}' already exists. Files may be overwritten.")
        # return 0

    # Split video
    print(f"Splitting video '{args.video}' into frames...")
    split_video(args.video, outdir, args.spilt_fps)


def split_video(video_path: str, outdir: str, spilt_fps: int = 1):
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = vc.get(cv2.CAP_PROP_FRAME_COUNT) / vc.get(cv2.CAP_PROP_FPS)
    print(f"  Video info: {video_path}")
    print(f"  -      fps: {fps:.2f}")
    print(f"  -    frame: {frame_count}")
    print(f"  -    width: {width}")
    print(f"  -   height: {height}")
    print(f"  - duration: {duration:.2f} s")

    file_base = os.path.splitext(os.path.basename(video_path))[0]

    if spilt_fps > int(fps):
        print(f"spilt_fps ({spilt_fps}) should be less than fps ({fps:.2f})")

    sample_interval = int(fps / spilt_fps + 0.5)
    if sample_interval < 1:
        sample_interval = 1

    print(f"  - sample interval: {sample_interval}")

    fid = 0
    split_cnt = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        if fid % sample_interval == 0:
            fn = os.path.join(outdir, f"{file_base}-{fid:04d}.jpg")
            cv2.imwrite(fn, frame)
            split_cnt += 1
        fid += 1


if __name__ == "__main__":
    main()
