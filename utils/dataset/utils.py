from typing import Callable, List
import os
import tqdm

from .types import ImageLabel
from .dataset_parser import VOCParser

from .variables import SUPPORTED_DATASET_FORMAT


def check_label_file(label_file_path: str, dataset_type: str = "VOC") -> bool:
    """
    Check if the label file exists
    """
    if dataset_type == "VOC":
        label_file_path = os.path.expandvars(
            os.path.expanduser(label_file_path)
        )
    else:
        raise NotImplementedError(
            f"Dataset type '{dataset_type}' not supported, "
            "choose from {SUPPORTED_DATASET_FORMAT}"
        )


def default_get_all_files(directory: str):
    file_paths: List[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file in [".DS_Store"]:
                continue
            file_paths.append(os.path.join(root, file))
    return file_paths


def get_all_label_files(
    datadir: str,
    dataset_type: str = "VOC",
    custom_get_all_files: Callable[[str], List[str]] = default_get_all_files,
):
    """
    获取指定目录下的所有已经标注的图片文件，根据 xml 便利
    """
    if not os.path.exists(datadir):
        raise FileNotFoundError(f"Directory '{datadir}' not found")

    file_list = custom_get_all_files(datadir)

    # filter out the xml files
    label_files: List[ImageLabel] = []
    pbar = tqdm.tqdm(file_list)
    for file_path in pbar:
        pbar.set_description(f"{file_path}")
        file_suffix = os.path.splitext(file_path)[-1]

        if dataset_type == "VOC":
            if file_suffix != ".xml":
                continue
            parser = VOCParser(label_file=file_path)
        else:
            raise NotImplementedError(
                f"Dataset type '{dataset_type}' not supported, "
                "choose from {SUPPORTED_DATASET_FORMAT}"
            )
        label_files.append(parser.data)

    return label_files
