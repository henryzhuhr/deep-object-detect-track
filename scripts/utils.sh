
video_list=(
    "~/data/medicinebox_dataset/videos/240609_in_1.mp4"
    "~/data/medicinebox_dataset/videos/240609_in_2.mp4"
    "~/data/medicinebox_dataset/videos/240609_in_3.mp4"
    "~/data/medicinebox_dataset/videos/240609_out_1.mp4"
    "~/data/medicinebox_dataset/videos/240609_out_2.mp4"
    "~/data/medicinebox_dataset/videos/240609_out_3.mp4"
    "~/data/medicinebox_dataset/videos/240610_dormitory_1.mp4"
    "~/data/medicinebox_dataset/videos/240610_dormitory_2"
    "~/data/medicinebox_dataset/videos/240610_lab_1.mp4"
)
for video in "${video_list[@]}"; do
    python utils/split-video.py -v $video --spilt-fps 2
done



image_dirs=(
    "~/data/medicinebox_dataset/240609_in_1"
    "~/data/medicinebox_dataset/240609_in_2"
    "~/data/medicinebox_dataset/240609_in_3"
    "~/data/medicinebox_dataset/240609_out_1"
    "~/data/medicinebox_dataset/240609_out_2"
    "~/data/medicinebox_dataset/240609_out_3"
    "~/data/medicinebox_dataset/240610_dormitory_1"
    "~/data/medicinebox_dataset/240610_dormitory_2"
    "~/data/medicinebox_dataset/240610_lab_1"
)
for image_dir in "${image_dirs[@]}"; do
    python auto_label.py \
        -d $image_dir \
        -c ~/data/medicinebox_dataset-organized/dataset.yaml \
        -m tmp/train/240611-第二次-m6/weights/best.engine \
        -s 1280
done


python dataset-process.py -d ~/data/medicinebox_dataset