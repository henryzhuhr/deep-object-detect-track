export PROJECT_HOME=$(pwd)
export PROJECT_NAME=$(basename $PROJECT_HOME)
export ENV_NAME=$(echo $PROJECT_NAME | tr '[:upper:]' '[:lower:]')
export ENV_PATH=./.env/$ENV_NAME


export OPENVINO_HOME=/opt/intel/openvino_2024
if [ -d "$OPENVINO_HOME" ]; then
    source $OPENVINO_HOME/setupvars.sh
fi

unset OPENVINO_HOME



# =============== Color Print Util ===============
DEFAULT=$(echo -en '\033[0m')
RED=$(echo -en '\033[00;31m')
GREEN=$(echo -en '\033[00;32m')
YELLOW=$(echo -en '\033[00;33m')
BLUE=$(echo -en '\033[00;34m')
MAGENTA=$(echo -en '\033[00;35m')
PURPLE=$(echo -en '\033[00;35m')
CYAN=$(echo -en '\033[00;36m')
LIGHTGRAY=$(echo -en '\033[00;37m')
LRED=$(echo -en '\033[01;31m')
LGREEN=$(echo -en '\033[01;32m')
LYELLOW=$(echo -en '\033[01;33m')
LBLUE=$(echo -en '\033[01;34m')
LMAGENTA=$(echo -en '\033[01;35m')
LPURPLE=$(echo -en '\033[01;35m')
LCYAN=$(echo -en '\033[01;36m')
WHITE=$(echo -en '\033[01;37m')

function print_info {
    echo -e "${CYAN}  - [INFO] $1${DEFAULT}"
}

function print_success {
    echo -e "${GREEN}  - [SUCCESS] $1${DEFAULT}"
}

function print_warning {
    echo -e "${YELLOW}  - [WARNING] $1${DEFAULT}"
}

function print_error {
    echo -e "${RED}  - [ERROR] $1${DEFAULT}"
}

