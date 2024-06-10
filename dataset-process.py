from typing import List, Set
import os
import argparse
import shutil
import yaml

from utils.dataset import get_all_label_files


class DatasetProcessArgs:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        # fmt: off
        parser.add_argument("-d", "--datadir", type=str, default="~/data/drink", help="Directory of the dataset")
        parser.add_argument("-s", "--savedir", type=str, default=None, help="Directory to save the organized dataset, default is '<datadir>-organized'")
        # fmt: on
        return parser.parse_args()

    def __init__(self) -> None:
        args = self.get_args()
        self.datadir: str = os.path.expandvars(os.path.expanduser(args.datadir))
        if self.datadir[-1] == "/" or self.datadir[-1] == "\\":
            self.datadir = self.datadir[:-1]

        if not os.path.exists(self.datadir):  # check if the directory exists
            raise FileNotFoundError(
                f"Directory '{self.datadir}' not found, check '--datadir {args.datadir}'"
            )

        if args.savedir is None:
            self.savedir: str = f"{self.datadir}-organized"
            print(f"Save directory is not specified, save to '{self.savedir}'")
        else:
            self.savedir: str = os.path.expandvars(
                os.path.expanduser(args.savedir)
            )


def main():
    args = DatasetProcessArgs()

    # -- get all label files, type: List[ImageLabel]
    label_file_list = get_all_label_files(args.datadir)

    # -- organize the dataset into a new directory
    organized_datadir = args.savedir
    if not os.path.exists(organized_datadir):
        os.makedirs(organized_datadir, exist_ok=False)
    else:
        raise FileExistsError(
            f"Directory '{organized_datadir}' already exists."
            f"\033[00;33m To avoid overwriting, please manually delete by\033[0m"
            f"\033[00;32m 'rm -rf {organized_datadir}'\033[0m"
            f"\033[00;33m and run this script again.\033[0m"
        )

    images_dir = os.path.join(organized_datadir, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels_dir = os.path.join(organized_datadir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # -- get all classes first to order classes
    class_set: Set[str] = set()
    for label_file in label_file_list:
        for cls in label_file.get_all_class():
            class_set.add(cls)
    class_map = {cls: i for i, cls in enumerate(sorted(list(class_set)))}

    with open(os.path.join(organized_datadir, "classes.txt"), "w") as f:
        for icls, cls in enumerate(class_map):
            f.write(f"{icls} {cls}\n")

    # -- copy image files and create label files
    image_file_list: List[str] = []
    for label_file in label_file_list:
        src_image_file = os.path.join(
            (label_file.parent_dir), label_file.image_file
        )
        dst_image_file = os.path.join(
            images_dir, dst_file_name := os.path.basename(label_file.image_file)
        )
        shutil.copy(src_image_file, dst_image_file)
        image_file_list.append(dst_file_name)

        obj_list = label_file.to_coco()
        label_file_name = os.path.splitext(label_file.label_file)[0] + ".txt"
        with open(os.path.join(labels_dir, label_file_name), "w") as f:
            for obj in obj_list:  # [cls_name, x, y, w, h]
                cls_name, *bbox = obj
                cls_id = class_map[cls_name]
                bbox_str = " ".join([str(x)[:8] for x in bbox])
                f.write(f"{cls_id} {bbox_str}\n")

    # -- create train.txt and dataset.yaml which are used in training
    with open(os.path.join(organized_datadir, "train.relpath.txt"), "w") as f:
        for image_file in image_file_list:
            f.write(f"images/{image_file}\n")

    with open(os.path.join(organized_datadir, "train.txt"), "w") as f:
        for image_file in image_file_list:
            f.write(f"{organized_datadir}/images/{image_file}\n")

    with open(os.path.join(organized_datadir, "dataset.yaml"), "w") as f:
        data_dict = {
            "path": organized_datadir,
            "train": "./train.txt",
            "val": "./train.txt",
            "names": {i: cls for cls, i in class_map.items()},
        }
        yaml.dump(data_dict, f, sort_keys=False)


if __name__ == "__main__":
    main()
