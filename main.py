import cv2

from modules.detector.tensort_detector_v10 import TensorRTDetectorV10


def main():
    detector = TensorRTDetectorV10()
    detector.load_model("~/.cache/ultralytics/yolo11n.engine", verbose=True)

    img_path = "images/bus.jpg"

    img = cv2.imread(img_path)


if __name__ == "__main__":
    main()
