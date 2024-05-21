# -- Uncomment and set to the desired Python version --
# CUSTOM_PYTHON_VERSION=3.10

source scripts/base.sh

export ENV_PATH=$ENV_PATH.venv

if [ ! -d $ENV_PATH ]; then
    python3 -m venv $ENV_PATH
    print_success "Create Python environment in '$ENV_PATH'"
else
    print_success "Python environment '$ENV_PATH' already exists."
fi

source $ENV_PATH/bin/activate

source scripts/python-install-requirements.sh