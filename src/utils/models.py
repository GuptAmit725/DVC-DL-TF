import tensorflow as tf
import logging

def get_VGG16_model(input_shape=list,
                    model_path=str
                    ) -> tf.python.keras.engine.functional.Functional:
    model = tf.keras.applications.vgg16(input_shape,
                                        weights='imagenet',
                                        include_top = False
                                        )
    model.save(model_path)
    logging.info(f"VGG16 is successfully loaded and saved at {model_path}")
    return model

def prepare_full_model():
    return