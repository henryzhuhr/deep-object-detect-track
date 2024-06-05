if [ ! -f "scripts/base.sh" ]; then
    echo "scripts/base.sh not found"
    exit 1
fi
source scripts/base.sh


# =============== Set Training Variables ================

DATASET_DIR=~/data/yolodataset
TRAIN_DATA_DIR=$DATASET_DIR-organized

# if GPU memory is enough, recommended training with Single but not Multiple GPU
TRAIN_DEVICE="0,1" # Multiple GPUs
TRAIN_DEVICE="0"   # Single GPU
TRAIN_DEVICE="cpu" # default, MacOS or without GPU

MODEL_NAME=yolov5s
PRETRAINED_MODEL=$CACHE_DIR/yolov5/yolov5s.pt

BATCH_SIZE=4
EPOCHS=4

# =======================================================

print_info "Prepare Training the model ..."

YOLOV5_PROJECT_HOME=$PROJECT_HOME/projects/yolov5

print_info "Checking YOLOv5 project ..."
if [ ! -d "$YOLOV5_PROJECT_HOME" ]; then
    print_error "YOLOv5 project not found at $YOLOV5_PROJECT_HOME"
    print_tip "Please run 'git submodule update --init --recursive'"
    exit 1
else
    print_success "YOLOv5 project found in '$YOLOV5_PROJECT_HOME'"
fi

cd $YOLOV5_PROJECT_HOME

print_info "Start Training the model ..."

python3 train.py \
    --data $TRAIN_DATA_DIR/dataset.yaml \
    --cfg models/$MODEL_NAME.yaml \
    --weights $PRETRAINED_MODEL \
    --epochs $EPOCHS \
    --batch $BATCH_SIZE \
    --device $TRAIN_DEVICE \
    --workers 8 \
    --project $PROJECT_HOME/tmp/runs/train

# python train.py --data $env:USERPROFILE/data/bottle-organized/dataset.yaml  --cfg models/yolov5s.yaml --weights "../../.cache/yolov5/yolov5s.pt" --epochs 10 --batch 4 --device 0