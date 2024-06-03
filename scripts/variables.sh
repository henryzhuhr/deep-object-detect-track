
# =============== Environment Variables ================
# -- install python in `user` or `project` level
# export BASE_ENV_PATH=$HOME    # --    user level
export BASE_ENV_PATH=.          # -- project level

# ================== Project Variables ==================
export PROJECT_HOME=$(pwd)
export PROJECT_NAME=$(basename $PROJECT_HOME)
DEFAULT_ENV_NAME=$(echo $PROJECT_NAME | tr '[:upper:]' '[:lower:]')

export ENV_NAME=$([ -z "$ENV_NAME" ] && echo $DEFAULT_ENV_NAME || echo $ENV_NAME)
# export ENV_NAME="" # -- Uncomment to customize the environment name

export ENV_PATH=$BASE_ENV_PATH/.env/$ENV_NAME

# ================== Project Variables ==================
export CACHE_DIR=$PROJECT_HOME/.cache


# ================== Python Variables ==================
# CUSTOM_PYTHON_VERSION=3.12    # -- Uncomment and set to the desired Python version (only for conda)
export PIP_QUIET=true           # -- pip install with quiet/verbose

# ================== Enabling CUDA ==================
# -- Variables related to CUDA should be written to ~/.bashrc instead of here
# -- Uncomment to Overwrite the CUDA version if needed
# export CUDA_VERSION=12.1
# export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
# export PATH="$CUDA_HOME/bin:$PATH"

# ================== Enabling OpenVINO ==================
export OPENVINO_HOME=/opt/intel/openvino_2024
if [ -d "$OPENVINO_HOME" ]; then
    source $OPENVINO_HOME/setupvars.sh
fi
unset OPENVINO_HOME

