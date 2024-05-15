# -- Uncomment and set to the desired Python version --
# CUSTOM_PYTHON_VERSION=3.10 

source scripts/base.sh

export ENV_PATH=$ENV_PATH.conda

if [ ! -d $ENV_PATH ]; then
    SYS_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    if [ -z "$CUSTOM_PYTHON_VERSION" ]; then
        ENV_PYTHON_VERSION=$SYS_PYTHON_VERSION
    else
        ENV_PYTHON_VERSION=$CUSTOM_PYTHON_VERSION
    fi
    conda create -p $ENV_PATH -y python=$ENV_PYTHON_VERSION 
    print_success "Create Python environment in '$ENV_PATH'"
else
    print_success "Conda environment '$ENV_PATH' already exists."
fi

eval "$(conda shell.$(basename $SHELL) hook)"
conda activate $ENV_PATH
print_success "Activated $(python --version) in ($ENV_PATH)"

source scripts/python-install-requirements.sh