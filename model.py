from get_config import config
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from tensorflow.keras.layers.experimental import preprocessing

img_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomContrast(factor=0.20)
    ],name="Augmentation")


def generate_weights(data_counters, extra_weight=False, ew_value=1):
    count_covid = data_counters[config.content['CLASSES'][0]]
    count_non_covid = data_counters[config.content['CLASSES'][1]]

    total_COUNT = count_covid + count_non_covid

    weight_for_0 = (1 / count_covid) * total_COUNT / 2.0
    weight_for_1 = (1 / count_non_covid) * total_COUNT / 2.0

    if extra_weight:
        weight_for_1 *= ew_value

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print(f'Weight for class 0: {weight_for_0}')
    print(f'Weight for class 1: {weight_for_1}')

    return class_weight


def build_model(OPTIMIZER, LOSS, METRICS, NETWORK):
    model = None
    inputs = layers.Input(shape=config.content['input_shape'])
    x = img_augmentation(inputs)
    baseModel = NETWORK(include_top=False, input_tensor=x, weights="imagenet", pooling='avg')

    baseModel.trainable = False

    x = BatchNormalization(axis=-1, name="Batch-Normalization-1")(baseModel.output)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization(axis=-1, name="Batch-Normalization-2")(x)
    x = Dropout(.2, name="Dropout-1")(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization(axis=-1, name="Batch-Normalization-3")(x)

    outputs = Dense(len(config.content['CLASSES']), activation="softmax", name="Classifier")(x)
    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs, name="Deep-COVID")

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    return model


def fit_model(Epochs, Callbacks, model, train_ds, val_ds, class_weight=None):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Epochs,
        callbacks=Callbacks,
        verbose=1,
        class_weight=class_weight
    )
    return history


def evaluate_model(model, test_ds):
    results = model.evaluate(test_ds, return_dict=True)
    print('\nModel Evaluation:')
    print(results['accuracy'] * 100)
    return results
