import tensorflow as tf
import logging

def get_VGG16_model(input_shape:list,
                    model_path:str
                    ) -> tf.keras.models.Model:
    model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
                                        weights='imagenet',
                                        include_top = False
                                        )
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

    full_model.summary()
    return full_model