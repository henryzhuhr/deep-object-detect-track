import argparse
from ultralytics import YOLO

from aux import validate_file_exist


class DetectArgs:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="https://docs.ultralytics.com/modes/train/#train-settings")
        # fmt: off
        parser.add_argument("--weights", type=str, default=".cache/yolov8/yolov8s.pt")
        parser.add_argument("--cfg", type=str, default="yolov8s")
        parser.add_argument("--data", type=str, default=".cache/data/drink-organized/dataset.yaml")
        parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
        parser.add_argument("--batch", type=int, default=-1, help="-1 for autobatch")
        parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640)
        parser.add_argument("--rect", action="store_true", help="rectangular training")
        parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
        parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
        parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
        parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
        parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
        parser.add_argument("--project", default= "tmp/detect", help="save to project/name")
        parser.add_argument("--name", default="train", help="save to project/name")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        # fmt: on
        return parser.parse_args()

    def __init__(self) -> None:
        args = self.get_args()
        self.args = args
        self.weights = args.weights
        self.cfg = f"{args.cfg}.yaml"
        self.data = args.data
        self.epochs = args.epochs
        self.batch = args.batch
        self.imgsz = args.imgsz
        self.rect = args.rect
        self.resume = args.resume
        self.cache = args.cache
        self.device = args.device
        self.multi_scale = args.multi_scale
        self.workers = args.workers
        self.project = args.project
        self.name = args.name
        self.exist_ok = args.exist_ok
        self.device = args.device


def main():
    # args = DetectArgs().get_args()
    args = DetectArgs()

    # Load a COCO-pretrained YOLOv8n model
    model = YOLO(args.cfg).load(args.weights)

    # Train the model with 2 GPUs
    results = model.train(
        # weights=args.weights,
        # cfg=args.cfg,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        rect=args.rect,
        resume=args.resume,
        cache=args.cache,
        device=args.device,
        multi_scale=args.multi_scale,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
