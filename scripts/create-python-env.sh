if [ ! -f "scripts/base.sh" ]; then
    echo "scripts/base.sh not found"
    exit 1
fi
source scripts/base.sh

# Set default values
env_provider="venv"
install=true


while [ $# -gt 0 ]
do
key="$1"
case $key in
    -e|--env-provider)
    env_provider="$2"
    shift # shift past argument
    shift # shift past value
    ;;
    -ni|--no-install)
    install=false
    shift # shift past argument
    ;;
    -h|--help)
    echo "Usage: create-python-env.sh [OPTIONS]"
    echo "Options:"
    echo "  -e, --env-provider <provider>  Set the environment provider (venv or conda)"
    echo "  -ni, --no-install              Skip Python requirements installation"
    echo "  -h, --help                     Show this help message and exit"
    exit 0
    ;;
    *)
    # unknown option
    echo "Unknown option: $key"
    exit 1
    ;;
esac
done

print_info "Environment Provider set to: $env_provider"

if [ "$env_provider" = "venv" ]; then
    run_script "scripts/utils/python-activate.venv.sh"
elif [ "$env_provider" = "conda" ]; then
    run_script "scripts/utils/python-activate.conda.sh"
else
    print_error "Invalid environment provider: $env_provider , exiting..."
    print_tip "Valid options are: 'venv' or 'conda'"
    exit 1
fi

if [ "$install" = true ]; then
    print_info "Installing Python requirements..."
    run_script "scripts/utils/python-install-requirements.sh"
else
    print_warning "Skipping Python requirements installation..."
fi

print_activate_env_message

