import os
import argparse
from typing import List

import yaml


class TestArgs:
    def __init__(self) -> None:
        args = self.get_args()
        self.dataset_config_file: str = os.path.expandvars(
            os.path.expanduser(args.dataset_config)
        )

        # check if the directory exists
        if not os.path.exists(self.dataset_config_file):
            raise FileNotFoundError(
                f"Dataset configuration file not found: {self.dataset_config_file}"
            )

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


def main():
    args = TestArgs()

    dataset_config_file = args.dataset_config_file
    with open(dataset_config_file, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    print(data_config)
    valid_list_file = data_config["val"]
    train_list_file = data_config["train"]
    test_list:List[str]=None
    if os.path.exists(valid_list_file):

        
    

if __name__ == "__main__":
    main()
