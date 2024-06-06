# 启动 PowerShell：打开 PowerShell 控制台。可以通过在Windows搜索框中键入“PowerShell”并选择相应的选项来启动。

# 设置执行策略（可选）：如果您的系统的执行策略不允许运行脚本，您可能需要先更改执行策略。可以通过在管理员权限下打开的 PowerShell 中运行以下命令来更改执行策略：

# powershell
# 复制代码
# Set-ExecutionPolicy RemoteSigned
# 这将允许运行本地的、来自 Internet 的已签名脚本，但不会执行未签名的本地脚本。

# Check if scripts\base.ps1 exists
if (-not (Test-Path "scripts\base.ps1")) {
    Write-Host "scripts\base.ps1 not found"
    exit 1
}

. "scripts\base.ps1"

# Check if ENV_PATH is set
if (-not $script:ENV_PATH) {
    print_error "ENV_PATH is not set. Please set it in 'scripts\variables.ps1'"
    exit 1
}

$script:ENV_PATH = "$script:ENV_PATH.conda"

# Check if ENV_PATH directory exists
if (-not (Test-Path $script:ENV_PATH -PathType Container)) {
    $DEFAULT_PYTHON_VERSION = 3.10
    if (-not $script:CUSTOM_PYTHON_VERSION) {
        $ENV_PYTHON_VERSION = $DEFAULT_PYTHON_VERSION
    } else {
        $ENV_PYTHON_VERSION = $script:CUSTOM_PYTHON_VERSION
    }
    conda create -p $script:ENV_PATH -y python=$ENV_PYTHON_VERSION
    print_success "Create Python environment in '$script:ENV_PATH'"
} else {
    print_info "Conda environment '$script:ENV_PATH' already exists."
}

conda activate $script:ENV_PATH

# 该安装命令参考 https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r projects/yolov5/requirements.txt
pip install -r requirements/requirements.train.txt

conda deactivate

function print_activate_env_message {
    Write-Host ""
    print_info "Run command below to activate the environment:"
    Write-Host ""
    Write-Host "source ~/.$(Split-Path -Leaf $PROFILE)"
    Write-Host "conda activate $script:ENV_PATH"
    Write-Host ""
    print_info "Then run command below to deactivate the environment:"
    Write-Host ""
    Write-Host "source ~/.$(Split-Path -Leaf $PROFILE)"
    Write-Host "conda deactivate"
    Write-Host ""
}
print_activate_env_message

# cd .\project\deep-object-detect-track\
# conda activate .\.env\deep-object-detect-track.conda\
# python .\export.py --weights "C:\Users\29650\project\deep-object-detect-track\.cache\yolov5\yolov5s.pt" --data data/coco128.yaml --simplify --include onnx openvino engine --device 0
# python .\export.py --weights "C:\Users\29650\project\deep-object-detect-track\.cache\yolov5\yolov5s.pt" --data data/coco128.yaml --include onnx openvino engine --device 0
# python .\export.py --weights "C:\Users\29650\project\deep-object-detect-track\.cache\yolov5\yolov5s.pt" --data data/coco128.yaml --include engine --device 0