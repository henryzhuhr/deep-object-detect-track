import os
from pathlib import Path


def validate_file_exist(file_path: str):
    fp: str | Path = os.path.expandvars(os.path.expanduser(file_path))
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File not found: {fp}")
    return fp
