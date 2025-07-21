import cv2


def check_opencv():
    try:
        # Check if OpenCV is installed and can be imported
        version = cv2.__version__
        print(f"OpenCV version: {version}")

        # Check if the 'cv2' module has the 'dnn' submodule
        if hasattr(cv2, "dnn"):
            print("OpenCV DNN module is available.")
        else:
            print("OpenCV DNN module is not available.")

    except ImportError:
        print(
            "OpenCV is not installed. Please install it using 'pip install opencv-python'."
        )
    except Exception as e:
        print("An error occurred while checking OpenCV:")
        print(e)


if __name__ == "__main__":
    check_opencv()
