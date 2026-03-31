import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal, Optional, Union

import huggingface_hub as hf
import kagglehub
import pandas as pd
import requests

RANDOM_SEED: int = 42
DATASET_TYPE_FORMAT = Literal["csv", "excel", "json", "sql"]
PATH_TO_DATA_DIR: Path = Path(os.path.abspath(path=__file__)).parent / "datasets"
EXTENSIONS_DATA_FILES: list[str] = ["csv", "excel", "json", "sql_db"]
DATASETS_FOLDERS_MAP: dict[str, Union[Path | list[Path]]] = {}
DATASETS_REMOTE_PATHS_MAP: dict[str, tuple[str, str, str]] = {
    "csv": ("maharshipandya/spotify-tracks-dataset", "dataset.csv", "hf"),  # repo_id, filename, type_api
    "json": ("rtatman/iris-dataset-json-version", "iris.json", "kh"),  # repo_id, filename, type_api
    "excel": ("rohanmistry231/Practice-Datasets-for-Excel/main/Avacado%20Prices", "avocado.xlsx", "gh"),  # repo_url, filename, type_api
    "sql_db": ("sql_sample_db", "IRIS_DATASET_TABLE", "sql_db"),  # db_name, table_name, type_api
}


def make_data_dir(subset_data_folders: list[str] = EXTENSIONS_DATA_FILES) -> None:
    PATH_TO_DATA_DIR.mkdir(exist_ok=True)
    
    for extension in subset_data_folders:
        extension_dir: Path = PATH_TO_DATA_DIR / extension
        extension_dir.mkdir(exist_ok=True)
        
        DATASETS_FOLDERS_MAP[extension] = extension_dir
        print(f"{extension} file datasets located at: {extension_dir}")


def download_hf_dataset(remote_access_tuple: tuple, output_dir: Path) -> None:
    """Download dataset from Hugging Face Hub."""
    hf.hf_hub_download(repo_id=remote_access_tuple[0], filename=remote_access_tuple[1], repo_type="dataset",  local_dir=output_dir)


def download_kh_dataset(remote_access_tuple: tuple, output_dir: Path) -> None:
    """Download dataset from KaggleHub."""
    kagglehub.dataset_download(handle=remote_access_tuple[0], path=remote_access_tuple[1], output_dir=output_dir)


def download_gh_dataset(remote_access_tuple: tuple, output_dir: Path) -> None:
    """Download dataset directly from GitHub."""
    url = f"https://raw.githubusercontent.com/{remote_access_tuple[0]}/{remote_access_tuple[1]}"
    response = requests.get(url=url)
    output_file_path = output_dir / Path(remote_access_tuple[1]).name
    
    output_dir.mkdir(exist_ok=True)
    
    if response.status_code == 200:
        with open(output_file_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"GitHub responded with: HTTP {response.status_code} - {response}")


def __create_table(cursor: sqlite3.Cursor, table_name: str) -> None:
    def __convert_to_sqlite_type(py_val: Any) -> str:
        if isinstance(py_val, bool):
            return "INTEGER"
        if isinstance(py_val, int):
            return "INTEGER"
        if isinstance(py_val, float):
            return "REAL"
        if isinstance(py_val, str):
            return "TEXT"
        if py_val is None:
            return "NULL"
        return "TEXT"
    
    json_dataset_path: Path = DATASETS_FOLDERS_MAP.get("json", None)
    if not json_dataset_path:
        print("JSON Dataset path not found.")
        return

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    with open(json_dataset_path / "iris.json", "r", encoding="utf-8") as f:
        iris_dataset_json: list[dict] = json.load(fp=f)
    
    first_elem: dict = iris_dataset_json[0]
    columns: list[str] = list(first_elem.keys())
    sqlite_types: list[str] = [
        __convert_to_sqlite_type(val) for val in first_elem.values()
    ]
    zipped_col_vals = dict(zip(columns, sqlite_types))
    table_def: list[str] = ["id INTEGER PRIMARY KEY AUTOINCREMENT"] + [" ".join([col_name, col_type]) for col_name, col_type in zipped_col_vals.items()]
    create_table_sql_query: str = f"CREATE TABLE IF NOT EXISTS {table_name}({",".join(table_def)})"
    
    cursor.execute(create_table_sql_query)


def __populate_table(cursor: sqlite3.Cursor, table_name: str) -> None:
    json_dataset_path: Path = DATASETS_FOLDERS_MAP.get("json", None)
    if not json_dataset_path:
        print("JSON Dataset path not found.")
        return

    with open(json_dataset_path / "iris.json", "r", encoding="utf-8") as f:
        iris_dataset_json: list[dict] = json.load(fp=f)
    
    first_elem: dict = iris_dataset_json[0]
    columns: list[str] = list(first_elem.keys())
    
    sql_query: list[str] = f"INSERT INTO {table_name} ({",".join(columns)}) VALUES ({",".join(["?"] * len(columns))})"
    rows = [tuple(item[col] for col in columns) for item in iris_dataset_json]
    cursor.executemany(sql_query, rows)


def download_sql_dataset(remote_access_tuple: tuple, output_dir: Path) -> None:
    """Download or generate SQLite dataset."""
    output_dir.mkdir(exist_ok=True)
    db_path = output_dir / remote_access_tuple[0]
    connection_sqlite = None

    try:
        connection_sqlite = sqlite3.connect(database=db_path)
        cursor = connection_sqlite.cursor()

        __create_table(cursor, table_name=remote_access_tuple[1])
        connection_sqlite.commit()

        __populate_table(cursor, table_name=remote_access_tuple[1])
        connection_sqlite.commit()
    except sqlite3.Error as error:
        print(f"Error occurred: {error}")
    finally:
        if connection_sqlite:
            connection_sqlite.close()


def download_dataset(remote_access_tuple: str, output_dir: Path) -> None:
    match remote_access_tuple[2]:
        case "hf": 
            download_hf_dataset(remote_access_tuple=remote_access_tuple, output_dir=output_dir)
        case "kh":
            download_kh_dataset(remote_access_tuple=remote_access_tuple, output_dir=output_dir)
        case "gh":
            download_gh_dataset(remote_access_tuple=remote_access_tuple, output_dir=output_dir)
        case "sql_db":
            download_sql_dataset(remote_access_tuple=remote_access_tuple, output_dir=output_dir)
            

def create_additional_json_files() -> list[Path]:
    path_to_json_file: Path = __get_path_to_json_dataset()
    path_to_json_folder: Path = path_to_json_file.parent

    df: pd.DataFrame = pd.read_json(path_to_json_file, orient="records")

    orient_formats: list[str] = ["split", "index", "columns"]

    new_json_filepaths: dict[str, Path] = {"records": path_to_json_file}

    for orient in orient_formats:
        output_file: Path = path_to_json_folder / f"iris_{orient}.json"
        df.to_json(output_file, orient=orient, indent=2)
        
        new_json_filepaths[orient] = output_file
    
    jsonl_file: Path = path_to_json_folder / "iris_lines.jsonl"
    df.to_json(jsonl_file, orient="records", lines=True)
    new_json_filepaths["lines"] = jsonl_file
    
    return new_json_filepaths


def read_additional_json_files_paths() -> dict[str, Path]:
    path_to_json_file: Path = __get_path_to_json_dataset()
    path_to_json_folder: Path = path_to_json_file.parent

    json_filepaths: dict[str, Path] = {
        "records": path_to_json_file,
        "split": path_to_json_folder / "iris_split.json",
        "index": path_to_json_folder / "iris_index.json",
        "columns": path_to_json_folder / "iris_columns.json",
        "lines": path_to_json_folder / "iris_lines.jsonl"
    }
    
    return json_filepaths


def download_all_datasets() -> None:
    for extension, remote_access_tuple in DATASETS_REMOTE_PATHS_MAP.items():
        approp_local_path: Path = DATASETS_FOLDERS_MAP.get(extension, None)
        
        if not approp_local_path:
            approp_local_path: Path = PATH_TO_DATA_DIR / "unknown"
            approp_local_path.mkdir(exist_ok=True)
        
        download_dataset(remote_access_tuple=remote_access_tuple, output_dir=approp_local_path)
        

def __get_path_to_csv_dataset() -> Path:
    path_to_csv_folder: Path = DATASETS_FOLDERS_MAP.get("csv", None)
    data_tuple: tuple[str, str, str] = DATASETS_REMOTE_PATHS_MAP["csv"]
    
    if not path_to_csv_folder:
        print("No path to csv datasets folder found. Downloading dataset...")
        make_data_dir(subset_data_folders=["csv"])
        approp_local_path: Path = DATASETS_FOLDERS_MAP["csv"]
        
        download_dataset(remote_access_tuple=data_tuple, output_dir=approp_local_path)
        path_to_csv_folder = Path(approp_local_path)
    
    return path_to_csv_folder / data_tuple[1]


def __get_path_to_excel_dataset() -> Path:
    path_to_excel_folder: Path = DATASETS_FOLDERS_MAP.get("excel", None)
    data_tuple: tuple[str, str, str] = DATASETS_REMOTE_PATHS_MAP["excel"]
    
    if not path_to_excel_folder:
        print("No path to excel datasets folder found. Downloading dataset...")
        make_data_dir(subset_data_folders=["excel"])
        approp_local_path: Path = DATASETS_FOLDERS_MAP["excel"]
        
        download_dataset(remote_access_tuple=data_tuple, output_dir=approp_local_path)
        path_to_excel_folder = Path(approp_local_path)
    
    return path_to_excel_folder / data_tuple[1]


def __get_path_to_json_dataset() -> Path:
    path_to_json_folder: Path = DATASETS_FOLDERS_MAP.get("json", None)
    data_tuple: tuple[str, str, str] = DATASETS_REMOTE_PATHS_MAP["json"]
    
    if not path_to_json_folder:
        print("No path to json datasets folder found. Downloading dataset...")
        make_data_dir(subset_data_folders=["json"])
        approp_local_path: Path = DATASETS_FOLDERS_MAP["json"]
        
        download_dataset(remote_access_tuple=data_tuple, output_dir=approp_local_path)
        path_to_json_folder = Path(approp_local_path)
    
    return path_to_json_folder / data_tuple[1]


def __get_path_to_sql_dataset() -> Path:
    path_to_sql_folder: Path = DATASETS_FOLDERS_MAP.get("sql_db", None)
    data_tuple: tuple[str, str, str] = DATASETS_REMOTE_PATHS_MAP["sql_db"]
    
    if not path_to_sql_folder:
        print("No path to sql datasets folder found. Downloading dataset...")
        make_data_dir(subset_data_folders=["sql_db"])
        approp_local_path: Path = DATASETS_FOLDERS_MAP["sql_db"]
        
        download_dataset(remote_access_tuple=data_tuple, output_dir=approp_local_path)
        path_to_sql_folder = Path(approp_local_path)
    
    return path_to_sql_folder / data_tuple[0]


def format_path_to_file(path: Path) -> str:
    base = Path(__file__).resolve().parent
    rel = Path(os.path.relpath(path.resolve(), start=base))
    return "../" + rel.as_posix()


def get_path_to_dataset_by_type(dataset_type: DATASET_TYPE_FORMAT) -> Optional[Path]:
    match dataset_type:
        case "csv":
            return __get_path_to_csv_dataset()
        case "excel":
            return __get_path_to_excel_dataset()
        case "json":
            return __get_path_to_json_dataset()
        case "sql_db":
            return __get_path_to_sql_dataset()
        case _:
            print(f"Unknown dataset type: {dataset_type}")
            return None


def create_sqlite_conn(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(database=db_path)


def get_col_names_from_table(conn: sqlite3.Connection, table_name: str) -> list[str]:
    cursor = conn.cursor()
    
    cursor.execute(f"PRAGMA table_info({table_name});")
    
    columns = cursor.fetchall()
    cursor.close()
    
    return columns


def close_sqlite_conn(conn: sqlite3.Connection) -> None:
    if conn:
        conn.close()