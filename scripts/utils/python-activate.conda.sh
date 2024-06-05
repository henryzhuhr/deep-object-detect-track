
if [ -z $ENV_PATH ]; then
    print_error "ENV_PATH is not set. Please set it in 'scripts/base.sh'"
    exit 1
fi

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
print_success "Environment $(python --version) Activated from ($ENV_PATH)"

function print_activate_env_message {
    echo ""
    print_tip "Run command below to activate the environment:"
    echo ""
    echo "source ~/.`basename $SHELL`rc"
    echo "conda activate $ENV_PATH"
    echo ""
    print_tip "Then run command below to deactivate the environment:"
    echo ""
    echo "source ~/.`basename $SHELL`rc"
    echo "conda deactivate"
    echo ""
}