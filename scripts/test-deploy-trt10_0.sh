source ~/.bashrc

ENV_PATH=".env/deploy-trt_10_0.venv"
if [ ! -d $ENV_PATH ];then
    python3 -m venv $ENV_PATH
fi
source $ENV_PATH/bin/activate

pip install pyyaml opencv-python \
    tensorrt==10.0.1 \
    cuda-python==12.1.0
# python3 -m pip install --force-reinstall nvidia-cudnn-cu12==8.9.2.26

python3 infer.py --model .cache/yolov5/yolov5s-10_0.engine