import argparse
import os
import shutil
from datetime import time

from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.models import load_models, get_unique_path_to_save_model
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
        IMAGE_SIZE=params['IMAGE_SIZE'][0:-1],
        BATCH_SIZE = params['BATCH_SIZE'],
        do_data_augmentation = params['AUGMENTATION']
    )
    ##Get the callbacks
    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    ##training the model
    steps_per_epochs = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    model.fit(train_generator,
              validation_data=validation_generator,
              epochs=params['EPOCHS'],
              steps_per_epoch=steps_per_epochs,
              validation_steps=validation_steps,
              callbacks=callbacks
              )

    ##Save the model
    trained_model_dir = os.path.join(artifacts_dir, artifacts['TRAINED_MODEL_DIR'])
    model_file_path = get_unique_path_to_save_model(trained_model_dir)
    model.save(model_file_path)
    logging.info(f"The trained model is saved at:\n {model_file_path}")

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