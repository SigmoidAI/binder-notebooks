import json
import os
from pathlib import Path
from typing import Union

import kagglehub
import pandas as pd

RANDOM_SEED: int = 42
PATH_TO_DATA_DIR: Path = Path(os.path.abspath(path=__file__)).parent / "datasets"
DATASETS_FOLDERS_MAP: dict[str, Union[Path | list[Path]]] = {}
DATASETS_REMOTE_PATHS_MAP: dict[str, tuple[str, str, str]] = {
    "json": ("rtatman/iris-dataset-json-version", "iris.json", "kh"),  # repo_id, filename, type_api
}


def make_data_dir() -> None:
    PATH_TO_DATA_DIR.mkdir(exist_ok=True)
    
    extension_dir: Path = PATH_TO_DATA_DIR / "json"
    extension_dir.mkdir(exist_ok=True)
    
    DATASETS_FOLDERS_MAP["json"] = extension_dir
    print(f"JSON file datasets located at: {extension_dir}")


def download_json_dataset() -> None:
    """Download dataset from KaggleHub."""
    remote_access_tuple: tuple[str, str, str] = DATASETS_REMOTE_PATHS_MAP["json"]
    approp_local_path: Path = DATASETS_FOLDERS_MAP.get("json", None)
    
    if not approp_local_path:
        approp_local_path: Path = PATH_TO_DATA_DIR / "unknown"
        approp_local_path.mkdir(exist_ok=True)
    
    kagglehub.dataset_download(handle=remote_access_tuple[0], path=remote_access_tuple[1], output_dir=approp_local_path)


def __get_path_to_json_dataset() -> Path:
    path_to_json_folder: Path = DATASETS_FOLDERS_MAP.get("json", None)
    data_tuple: tuple[str, str, str] = DATASETS_REMOTE_PATHS_MAP["json"]
    
    if not path_to_json_folder:
        print("No path to json datasets folder found. Downloading dataset...")
        make_data_dir(subset_data_folders=["json"])
        approp_local_path: Path = DATASETS_FOLDERS_MAP["json"]
        
        download_json_dataset()
        path_to_json_folder = Path(approp_local_path)
    
    return path_to_json_folder / data_tuple[1]


def get_sample_dataframe() -> pd.DataFrame:
    return pd.read_json(path_or_buf=__get_path_to_json_dataset()).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def return_json_file_path() -> Path:
    return __get_path_to_json_dataset()


def return_json_file_orient_type() -> str:
    path: Path = return_json_file_path()
    if not path:
        return "UNKNOWN"

    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return "records"
    if isinstance(raw, dict):
        if set(raw.keys()) == {"schema", "data"}:
            return "table"
        if set(raw.keys()) == {"columns", "index", "data"}:
            return "split"
        # sample the first value to distinguish index vs columns vs values
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, dict):
            return "index"   # {row_label: {col: val}}
        if isinstance(first_val, list):
            return "columns" # {col: [val, ...]}
        return "values"

    return "UNKNOWN"


def format_path_to_file(path: Path) -> str:
    base = Path(__file__).resolve().parent
    rel = Path(os.path.relpath(path.resolve(), start=base))
    return "../" + rel.as_posix()


