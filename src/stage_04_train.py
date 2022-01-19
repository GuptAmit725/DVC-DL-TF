import argparse
import os
import shutil
from datetime import time

from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.models import load_models
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator


STAGE = "prepare_callback" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def train_model(config_path:str, params_path:str)->None:
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']

    ##Get the untrained model
    train_model_dir = os.path.join(artifacts_dir,artifacts['TRAINED_MODEL_DIR'])
    create_directories([train_model_dir])

    untrained_full_model_path = os.path.join(
        artifacts_dir,
        artifacts["BASE_MODEL_DIR"],
        artifacts["UPDATED_BASE_MODEL_NAME"]
    )

    model = load_models(untrained_full_model_path)

    ##Get the data to create the datagenerator
    train_generator, validation_generator = train_valid_generator(
        data_dir=artifacts['DATA_DIR'],
        IMAGE_SIZE=params['IMAGE_SIZE'],
        BATCH_SIZE = params['BATCH_SIZE'],
        do_data_augmentation = params['AUGMENTATION']
    )
    ##Get the callbacks
    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks

    ##training the model

    ##Save the model


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage 3 completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e