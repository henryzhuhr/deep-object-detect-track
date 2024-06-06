import os
import argparse


class AutoLabelArgs:

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-d", "--image-dir", type=str, default="~/data/drink.unlabel/cola"
        )
        parser.add_argument(
            "-d", "--image-dir", type=str, default="~/data/drink.unlabel/cola"
        )
        return parser.parse_args()

    def __init__(self) -> None:
        # fmt: off
        args = self.get_args()
        self.image_dir: str = os.path.expandvars(os.path.expanduser(args.image_dir))
        if not os.path.exists(self.image_dir): # check if the directory exists
            raise FileNotFoundError(f"Dataset directory not found: {self.image_dir}")
        # fmt: on


def main():
    args = AutoLabelArgs()

    image_dir = args.image_dir




if __name__ == "__main__":
    main()
