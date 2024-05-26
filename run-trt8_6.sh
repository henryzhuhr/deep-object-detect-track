source ~/.bashrc

ENV_PATH=".env/deploy-trt_10_0.venv"
if [ ! -d $ENV_PATH ];then
    python3 -m venv $ENV_PATH
fi
source $ENV_PATH/bin/activate
python3 -m pip install -r requirements-trt10_0.txt

# export TENSORRT_HOME="$HOME/program/TensorRT-8.6.1.6"
# export PATH="$TENSORRT_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$TENSORRT_HOME/lib:$LD_LIBRARY_PATH"

# python3 -m pip install $TENSORRT_HOME/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl


python3 infer.py