source ~/.$(basename $SHELL)rc

# =============== Color Print ===============
DEFAULT=$(echo -en '\033[0m')
RED=$(echo -en '\033[00;31m')
GREEN=$(echo -en '\033[00;32m')
YELLOW=$(echo -en '\033[00;33m')
CYAN=$(echo -en '\033[00;36m')

function print_base     { echo -e "$1- [$2] $3${DEFAULT}"; }
function print_info     { print_base "$CYAN"    "INFO"      "$1"; }
function print_tip      { print_base "$YELLOW"  "TIP"       "$1"; }
function print_success  { print_base "$GREEN"   "SUCCESS"   "$1"; }
function print_warning  { print_base "$YELLOW"  "WARNING"   "$1"; }
function print_error    { print_base "$RED"     "ERROR"     "$1"; }

function run_script {
    if [ ! -f "$1" ]; then
        echo "$1 not found"
        exit 1
    fi
    source $1
}

function copy_file_to_custom {
    source_file=$1
    f_without_suffix=$(echo $source_file | sed 's/\.sh//')
    custom_file="$f_without_suffix.custom.sh"
    if [ ! -f "$custom_file" ] && [ -f "$source_file" ] ; then
        cp $source_file $custom_file
        print_success "Copied '$source_file' to '$custom_file'"
    fi
}

# copy_file_to_custom "scripts/variables.sh"
# copy_file_to_custom "scripts/train.sh"
# copy_file_to_custom "scripts/export-yolov5.sh"

if [ -f "scripts/variables.custom.sh" ]; then
    run_script "scripts/variables.custom.sh"
    print_success "Loaded custom variables: 'scripts/variables.custom.sh'"
elif [ -f "scripts/variables.sh" ]; then
    run_script scripts/variables.sh
    print_success "Loaded default variables: 'scripts/variables.sh'"
else
    echo "'scripts/variables.custom.sh' or 'scripts/variables.sh' not found"
    exit 1
fi
