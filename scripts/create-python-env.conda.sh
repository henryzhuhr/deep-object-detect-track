function run_script {
    if [ ! -f "$1" ]; then
        echo "$1 not found"
        exit 1
    fi
    source $1
}

run_script "scripts/base.sh"
run_script "scripts/python-activate.conda.sh"
run_script "scripts/python-install-requirements.sh"

print_activate_env_message