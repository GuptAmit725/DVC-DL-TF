import tensorflow as tf
import logging
import io
import os
from src.utils.callbacks import get_timestamp

def load_models(model_path:str)->tf.keras.models.Model:
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Untrained model is read from : {model_path}")
    logging.info(f"Untrained full model summary is {get_model_summary(model)}")

    return model


def get_model_summary(model):
    with io.StringIO() as stream:
        model.summary(
            print_fn=lambda x : stream.write(f'{x}\n')
        )
        summary_str = stream.getvalue()
    return summary_str


def get_VGG16_model(input_shape:list,
                    model_path:str
                    ) -> tf.keras.models.Model:
    model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
                                        weights='imagenet',
                                        include_top = False
                                        )
    logging.info(f'full model summary {get_model_summary(model)}')
    model.save(model_path)
    logging.info(f"VGG16 is successfully loaded and saved at {model_path}")
    return model


def prepare_full_model( base_model,
                        lr,
                        CLASSES=2,
                        freeze_all=True,
                        freeze_till=None
):
    if freeze_all:
        for layer in base_model.layers:
            layer.trainable=False
    elif (freeze_till is not None) and (freeze_till>0):
        for layer in base_model.layers[:-freeze_till]:
            layer.trainable=False

    ##adding our layers to base model

    flatten_in = tf.keras.layers.Flatten()(base_model.output)
    prediction = tf.keras.layers.Dense(CLASSES, activation = 'softmax')(flatten_in)

    full_model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction)
    full_model.compile(
        optimizer = tf.keras.optimizers.Adam(lr),
        loss  = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    logging.info(f"Custom layers to base model added and compiled.")

    return full_model

def get_unique_path_to_save_model(
        trained_model_dir:str,
        model_name:str="model.h5")->str:

    timestamp = get_timestamp(model_name)
    unique_model_name = f"{timestamp}_.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)
    return unique_model_path