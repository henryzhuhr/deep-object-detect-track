import tensorrt


def check_tensorrt():
    try:
        # Check if TensorRT is available
        print("TensorRT version:", tensorrt.__version__)
        print("TensorRT is installed and available.")
    except ImportError as e:
        print("TensorRT is not installed or not available.")
        print(e)
    except Exception as e:
        print("An error occurred while checking TensorRT:")
        print(e)


if __name__ == "__main__":
    check_tensorrt()
