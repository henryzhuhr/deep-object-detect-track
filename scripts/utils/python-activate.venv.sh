
if [ -z $ENV_PATH ]; then
    print_error "ENV_PATH is not set. Please set it in 'scripts/base.sh'"
    exit 1
fi

export ENV_PATH=$ENV_PATH.venv

if [ ! -d $ENV_PATH ]; then
    python3 -m venv $ENV_PATH
    print_success "Create Python environment in '$ENV_PATH'"
else
    print_info "Python environment '$ENV_PATH' already exists."
fi

source $ENV_PATH/bin/activate
print_success "Environment $(python --version) Activated from '$ENV_PATH'"

function print_activate_env_message {
    echo ""
    print_tip "Run command below to activate the environment:"
    echo ""
    echo "source ~/.`basename $SHELL`rc"
    echo "source $ENV_PATH/bin/activate"
    echo ""

    print_tip "if you delete the environment, run command below:"
    echo ""
    echo "source ~/.`basename $SHELL`rc"
    echo "rm -rf $ENV_PATH"
    echo ""
}