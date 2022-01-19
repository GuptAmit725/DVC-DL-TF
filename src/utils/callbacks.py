import tensorflow as tf
import os
import joblib
import time
import logging



def get_timestamp(filename:str)->str:
    timestamp = time.asctime().replace(' ','_').replace(':','.')
    unique_name = f"{filename}_at_{timestamp}"
    return unique_name

def create_and_save_tb_callbacks(callbacks_dir:str, tensorboard_log_dir:str)->\
        None:
    unique_name = get_timestamp("tb_logs")
    tb_running_file_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=tb_running_file_dir)
    tb_callback_path = os.path.join(callbacks_dir,"tensorboard.cb.cb")
    print(tb_callback_path)
    joblib.dump(tensorboard_callbacks, tb_callback_path)

    logging.info(f"Tensorboard callback is saved at {tb_callback_path}")

def create_checkpointing_callbacks(callbacks_dir:str,checkpoint_dir:str)->None:
    checkpoint_file = os.path.join(checkpoint_dir,"ckpt.model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= checkpoint_file,
        save_best_only=True
    )
    ckpt_path = os.path.join(callbacks_dir, "tensorboard.cb.cb")
    joblib.dump(checkpoint_callback, ckpt_path)
    logging.info(f"Checkpoint callbacks are saved at {ckpt_path} as binay file.")