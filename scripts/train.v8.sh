if [ ! -f "scripts/base.sh" ]; then
    echo "scripts/base.sh not found"
    exit 1
fi
source scripts/base.sh


# =============== Set Device ================
# -- if GPU memory is enough, recommended training with Single but not Multiple GPU
# -- see issue if error: https://github.com/pytorch/pytorch/issues/110000

# -- Multiple GPUs (use all)
# export CUDA_VISIBLE_DEVICES="0,1"
# TRAIN_DEVICE="0,1"

# -- Multiple GPUs (use only one)
# export CUDA_VISIBLE_DEVICES=0
# export WORLD_SIZE=1
# TRAIN_DEVICE=1

# -- Single GPU
# TRAIN_DEVICE=0

# -- Apple Silicon
# TRAIN_DEVICE="mps"

# -- CPU
TRAIN_DEVICE="cpu"

# =======================================================

# =============== Set Training Variables ================

DATASET_DIR=~/data/drink
TRAIN_DATA_DIR=$DATASET_DIR-organized

MODEL_NAME=yolov8s
PRETRAINED_MODEL=$CACHE_DIR/yolov8/yolov8s.pt

BATCH_SIZE=-1
EPOCHS=80

# =======================================================

print_info "Prepare Training the model ..."

python train.py \
    --data $TRAIN_DATA_DIR/dataset.yaml \
    --cfg $MODEL_NAME --weights $PRETRAINED_MODEL \
    --rect --img-size 1280 \
    --epochs $EPOCHS --batch $BATCH_SIZE \
    --device $TRAIN_DEVICE --workers 128 --cache ram \
    --project $PROJECT_HOME/tmp/detect --name train