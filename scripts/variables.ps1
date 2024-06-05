# =============== Environment Variables ================
# -- install python in `user` or `project` level
$script:BASE_ENV_PATH = "."         # -- project level

# ================== Project Variables ==================
$script:PROJECT_HOME = (Get-Location)
$script:PROJECT_NAME = (Split-Path -Leaf $script:PROJECT_HOME)
$DEFAULT_ENV_NAME = $script:PROJECT_NAME.ToLower()
$script:ENV_NAME = $(if (-not $script:ENV_NAME) { $DEFAULT_ENV_NAME } else { $script:ENV_NAME })
$script:ENV_PATH = Join-Path -Path $script:BASE_ENV_PATH -ChildPath ".env\$script:ENV_NAME"

# ================== Project Variables ==================
$script:CACHE_DIR = Join-Path -Path $script:PROJECT_HOME -ChildPath ".cache"

# ================== Python Variables ==================
# CUSTOM_PYTHON_VERSION=3.12    # -- Uncomment and set to the desired Python version (only for conda)

# ================== Enable CUDA ==================
# -- Variables related to CUDA should be written to ~/.bashrc instead of here
# -- Uncomment to Overwrite the CUDA version if needed
# $script:CUDA_VERSION=12.1
# $script:CUDA_HOME="/usr/local/cuda-${Env:CUDA_VERSION}"
# $script:PATH="$Env:CUDA_HOME/bin;$Env:PATH"
