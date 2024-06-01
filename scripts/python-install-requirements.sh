
print_info "Using Python: $(which python3)"
print_info "Installing Python requirements..."

if [ -z "${PIP_QUIET}" ]; then
    PIP_QUIET=false
fi

if [ "$PIP_QUIET" = true ]; then
    PIP_QUIET_FLAG="-q"
else
    PIP_QUIET_FLAG=""
fi

print_info "Upgrading pip..."
python3 -m pip install $PIP_QUIET_FLAG --upgrade pip

# install pytorch according to the CUDA version
if [ ! -z "${CUDA_VERSION}" ]; then
    print_info "CUDA_VERSION is set to $CUDA_VERSION. Installing PyTorch with CUDA support."
    python3 -m pip install torch torchvision $PIP_QUIET_FLAG --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d '.')
else
    print_warning "CUDA_VERSION is not set. Installing CPU version of PyTorch."
    python3 -m pip install torch torchvision $PIP_QUIET_FLAG --index-url https://download.pytorch.org/whl/cpu
fi


print_info "Installing other requirements..."
mkdir -p .cache/yolov5
if [ ! -f ".cache/yolov5/requirements.txt" ]; then
    cp projects/yolov5/requirements.txt .cache/yolov5/requirements.txt
fi
python3 -m pip install $PIP_QUIET_FLAG -r .cache/yolov5/requirements.txt
python3 -m pip install $PIP_QUIET_FLAG -r requirements/requirements.train.txt

if [ ! -z "${CUDA_VERSION}" ]; then
    python3 -m pip install $PIP_QUIET_FLAG onnxruntime-gpu
else
    python3 -m pip install $PIP_QUIET_FLAG onnxruntime
fi

if [ -d "$INTEL_OPENVINO_DIR" ]; then
    print_info "OpenVINO found in $INTEL_OPENVINO_DIR"
    print_info "Installing OpenVINO Python requirements from $INTEL_OPENVINO_DIR/python/requirements.txt"
    python3 -m pip install $PIP_QUIET_FLAG -r $INTEL_OPENVINO_DIR/python/requirements.txt
else
    print_warning "OpenVINO not found, skipping OpenVINO Python requirements installation"
    # print_info "Installing OpenVINO Python requirements from pip"
    # python3 -m pip install $PIP_QUIET_FLAG openvino-dev
fi

# freeze the requirements 
# python3 -m pip list --format=freeze > requirements.version.txt
