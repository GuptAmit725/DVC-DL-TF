import os
import yaml
import logging
import time
import pandas as pd
import json
import shutil

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content

def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")


def copy_files(source_download_dir:str, local_data_dir:str)->None:
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)

    for file in list_of_files:
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)

        shutil.copy(src,dest)

    logging.info(f"Copying succeeded!")


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")