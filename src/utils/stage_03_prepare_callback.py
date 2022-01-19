import argparse
import os
import shutil
from datetime import time

from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.callbacks import create_and_save_tb_callbacks, create_checkpointing_callbacks


STAGE = "prepare_callback" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def prepare_callback(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts['TENSORBOARD_ROOT_LOG_DIR'])
    checkpoint_dir = os.path.join(artifacts_dir, artifacts['CHECKPOINT_DIR'])
    callback_dir = os.path.join(artifacts_dir, artifacts['CALLBACK_DIR'])
    create_directories([
        'tensorboard_log_dir',
        'checkpoint_dir',
        'callback_dir'
    ])
    #tensorboard callbacks
    create_and_save_tb_callbacks(callback_dir, tensorboard_log_dir)
    create_checkpointing_callbacks(checkpoint_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prepare_callback(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage 2 completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e