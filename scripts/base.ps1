# =============== Color Print ===============
$DEFAULT = [char]27 + '[0m'
$RED = [char]27 + '[00;31m'
$GREEN = [char]27 + '[00;32m'
$YELLOW = [char]27 + '[00;33m'
$CYAN = [char]27 + '[00;36m'

function print_base     { param($color, $type, $message) Write-Host -NoNewline "$color- [$type] $message"; Write-Host $DEFAULT }
function print_info     { param($message) print_base $CYAN      "INFO"      $message }
function print_tip      { param($message) print_base $YELLOW    "TIP"       $message }
function print_success  { param($message) print_base $GREEN     "SUCCESS"   $message }
function print_warning  { param($message) print_base $YELLOW    "WARNING"   $message }
function print_error    { param($message) print_base $RED       "ERROR"     $message }

function run_script {
    param($scriptPath)
    if (-not (Test-Path $scriptPath -PathType Leaf)) {
        Write-Host "$scriptPath not found"
        exit 1
    }
    . $scriptPath  # 指定 powershell 执行脚本
}

$variablesCustomPath = "scripts/variables.custom.ps1"
$variablesDefaultPath = "scripts/variables.ps1"

if (Test-Path $variablesCustomPath -PathType Leaf) {
    run_script $variablesCustomPath
    print_success "Loaded custom variables: '$variablesCustomPath'"
}
elseif (Test-Path $variablesDefaultPath -PathType Leaf) {
    run_script $variablesDefaultPath
    print_success "Loaded default variables: '$variablesDefaultPath'"
}
else {
    Write-Host "'$variablesCustomPath' or '$variablesDefaultPath' not found"
    exit 1
}
