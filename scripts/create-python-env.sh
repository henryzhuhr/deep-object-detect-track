if [ ! -f "scripts/base.sh" ]; then
    echo "scripts/base.sh not found"
    exit 1
fi
source scripts/base.sh

# Set default values
env_provider="venv"
install=false
verbose=false

while [ $# -gt 0 ]
do
key="$1"
case $key in
    -e|--env-provider)
    env_provider="$2"
    shift # shift past argument
    shift # shift past value
    ;;
    -i|--install)
    install=true
    shift # shift past argument
    ;;
    -v|--verbose)
    verbose=true
    shift
    ;;
    -h|--help)
    echo "Usage: create-python-env.sh [OPTIONS]"
    echo "Options:"
    echo "  -e, --env-provider <provider>  Set the environment provider (venv or conda)"
    echo "  -i, --install                  Install Python requirements"
    echo "  -v, --verbose                  Enable verbose mode"
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

print_info "Python environment provider set to: $env_provider"

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
    print_warning "Skipping Python requirements installation. If you want to install them, add the '-i' or '--install' flag."
fi

print_activate_env_message

