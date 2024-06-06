import os
import argparse
from typing import List

import yaml


class TestArgs:

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-d",
            "--dataset-config",
            type=str,
            default="~/data/drink-organized/dataset.yaml",
        )
        return parser.parse_args()

    def __init__(self) -> None:
        # fmt: off
        args = self.get_args()
        self.dataset_config_file: str = os.path.expandvars(os.path.expanduser(args.dataset_config))
        if not os.path.exists(self.dataset_config_file): # check if the directory exists
            raise FileNotFoundError( f"Dataset configuration file not found: {self.dataset_config_file}")
        # fmt: on


def main():
    args = TestArgs()

    dataset_config_file = args.dataset_config_file
    with open(dataset_config_file, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    print(data_config)
    valid_list_file = os.path.join(data_config["path"], data_config["val"])
    train_list_file = os.path.join(data_config["path"], data_config["train"])
    test_list: List[str] = None

    def read_file(file_path: str) -> List[str]:
        fl: List[str] = []
        with open(file_path, "r") as f:
            for line in f:
                fl.append(line.strip())
        return fl

    if os.path.exists(valid_list_file):
        test_list = read_file(valid_list_file)
    elif os.path.exists(train_list_file):
        test_list = read_file(valid_list_file)
    else:
        raise FileNotFoundError(
            f"No valid('{valid_list_file}') or train('{train_list_file}') files found"
        )

    print(test_list)
    


if __name__ == "__main__":
    main()
