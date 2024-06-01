source ~/.$(basename $SHELL)rc
# =============== Environment Variables ================
# install python in `user` or `project` level
# export BASE_ENV_PATH=$HOME
export BASE_ENV_PATH=.

# export ENV_NAME=dodt # Uncomment if custom env name

# ================== Project Variables ==================
export PROJECT_HOME=$(pwd)
export PROJECT_NAME=$(basename $PROJECT_HOME)
DEFAULT_ENV_NAME=$(echo $PROJECT_NAME | tr '[:upper:]' '[:lower:]')
export ENV_NAME=$([ -z "$ENV_NAME" ] && echo $DEFAULT_ENV_NAME || echo $ENV_NAME)
export ENV_PATH=$BASE_ENV_PATH/.env/$ENV_NAME

# ================== Python Variables ==================
CUSTOM_PYTHON_VERSION=3.12    #  Uncomment and set to the desired Python version
export PIP_QUIET=true

# ================== Enabling OpenVINO ==================
export OPENVINO_HOME=/opt/intel/openvino_2024
if [ -d "$OPENVINO_HOME" ]; then
    source $OPENVINO_HOME/setupvars.sh
fi
unset OPENVINO_HOME

# =============== Color Print ===============
DEFAULT=$(echo -en '\033[0m')
RED=$(echo -en '\033[00;31m')
GREEN=$(echo -en '\033[00;32m')
YELLOW=$(echo -en '\033[00;33m')
CYAN=$(echo -en '\033[00;36m')

function print_base { echo -e "$1- [$2] $3${DEFAULT}"; }
function print_info { print_base "$CYAN" "INFO" "$1"; }
function print_success { print_base "$GREEN" "SUCCESS" "$1"; }
function print_warning { print_base "$YELLOW" "WARNING" "$1"; }
function print_error { print_base "$RED" "ERROR" "$1"; }



