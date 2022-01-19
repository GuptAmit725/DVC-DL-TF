import tensorflow as tf

def train_valid_generator(
        data_dir:str='data',
        IMAGE_SIZE:tuple=(224,224),
        BATCH_SIZE:int=32,
        do_data_augmentation:bool=True
)-> tuple:
    datagenerator_kawargs = dict(
        rescale = 1./255, validation_split=0.2
    )

    dataflow_kawargs = dict(
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        interpolation='bilinear'
    )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kawargs)
    valid_generator = valid_datagenerator.flow_from_directory(
        directory=data_dir,
        subset="validation",
        shuffle=False,
        **dataflow_kawargs
    )

    if do_data_augmentation:
        train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range = 40,
            horizontal_flip = True,
            width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2,
            **datagenerator_kawargs
        )
    else:
        train_data_generator = valid_datagenerator

    train_generator = train_data_generator.flow_from_directory(
        directory=data_dir, subset="training", shuffle=True,
        **dataflow_kawargs
    )

    return train_generator, valid_generator
