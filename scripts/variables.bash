#!/bin/bash

# =============== Environment Variables ================
# -- install python in `user` or `project` level
# export BASE_ENV_PATH=$HOME    # --    user level
export BASE_ENV_PATH=.          # -- project level


# ================== Project Variables ==================
export PROJECT_HOME=$(pwd)
export PROJECT_NAME=$(basename "$PROJECT_HOME")
DEFAULT_ENV_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]')

export ENV_NAME=$([ -z "$ENV_NAME" ] && echo "$DEFAULT_ENV_NAME" || echo "$ENV_NAME")
# export ENV_NAME="" # -- Uncomment to customize the environment name
export ENV_PATH=$BASE_ENV_PATH/.env/$ENV_NAME


# ================== Project Variables ==================
export CACHE_DIR="${HOME}/.cache"


# ================== Python Variables ==================
# CUSTOM_PYTHON_VERSION=3.12    # -- Uncomment and set to the desired Python version (only for conda)
export UV_DEFAULT_INDEX="https://mirrors.cloud.tencent.com/pypi/simple/"

# ================== Enable CUDA ==================
# -- Variables related to CUDA should be written to ~/.bashrc instead of here

# -- Optional: specify a specific CUDA version
# export CUDA_VERSION=12.1

if [[ -n "$CUDA_VERSION" ]]; then
    export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
else
    export CUDA_HOME="/usr/local/cuda"
fi
if [[ -d "$CUDA_HOME" ]]; then
    # Add bin to PATH (if not already there)
    if [[ ":$PATH:" != *":$CUDA_HOME/bin:"* ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
    fi

    # Add lib64 to LD_LIBRARY_PATH (prefer over lib)
    if [[ -d "$CUDA_HOME/lib64" ]]; then
        if [[ ":$LD_LIBRARY_PATH:" != *":$CUDA_HOME/lib64:"* ]]; then
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        fi
    elif [[ -d "$CUDA_HOME/lib" ]]; then
        if [[ ":$LD_LIBRARY_PATH:" != *":$CUDA_HOME/lib:"* ]]; then
            export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
        fi
    fi
else
    unset CUDA_HOME
fi

