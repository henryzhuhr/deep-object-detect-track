# -- Uncomment and set to the desired Python version --
# CUSTOM_PYTHON_VERSION=3.10 

PROJECT_HOME=$(pwd)
PROJECT_NAME=$(basename $PROJECT_HOME)
ENV_NAME=$(echo $PROJECT_NAME | tr '[:upper:]' '[:lower:]')
ENV_PATH=./.env/$ENV_NAME

if [ ! -d $ENV_PATH ]; then
    SYS_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    if [ -z "$CUSTOM_PYTHON_VERSION" ]; then
        ENV_PYTHON_VERSION=$SYS_PYTHON_VERSION
    else
        ENV_PYTHON_VERSION=$CUSTOM_PYTHON_VERSION
    fi
    conda create -p $ENV_PATH -y python=$ENV_PYTHON_VERSION 
else
    echo "Conda environment '$ENV_PATH' already exists."
fi



eval "$(conda shell.$(basename $SHELL) hook)"
conda activate $ENV_PATH
echo "Activated $(python --version) in ($ENV_PATH)"

# install pytorch according to the CUDA version
if [ ! -z "${CUDA_VERSION}" ]; then
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d '.')
else
    python3 -m pip install torch torchvision
    echo "CUDA_VERSION is not set. Installing CPU version of PyTorch."
fi
python3 -m pip install --upgrade pip

mkdir -p .cache/yolov5
if [ ! -f ".cache/yolov5/requirements.txt" ]; then
    cp projects/yolov5/requirements.txt .cache/yolov5/requirements.txt
fi
python3 -m pip install -r .cache/yolov5/requirements.txt
python3 -m pip install -r requirements.txt


# freeze the requirements 
# python3 -m pip list --format=freeze > requirements.version.txt