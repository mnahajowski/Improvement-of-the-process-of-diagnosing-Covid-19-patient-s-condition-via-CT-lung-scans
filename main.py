import os
import random
from zipfile import ZipFile

from batches_display import display_example_data
from import_data import load_default_dataset, get_filenames
import numpy as np
import math
import itertools
from tensorflow.keras.callbacks import Callback
from data_characteristics import characterize_data
from model import generate_weights, build_model, fit_model, evaluate_model
from prepare_data import prepare_datasets, convert_imgs, prepare_batches_for_training, dataset_to_numpy_util
from statistics import Plot_Learning_Curves, plot_confusion_matrix, display_other_metrics, generate_results_examples, \
    generate_confusion_matrixes


class CosineAnnealingScheduler(Callback):

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


SEED = 7
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)
import tensorflow.keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow.keras.backend as K

from tensorflow.keras.applications import NASNetLarge

NETWORK = NASNetLarge
from sklearn import metrics

try:
    tpu = None
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

print("Tensorflow version ", tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.config.optimizer.set_jit(True)

dataset_id1 = 'covid19-lung-ct-scans'
BATCH_SIZE = 4 * strategy.num_replicas_in_sync

CLASSES = ['COVID-19', 'Non-COVID-19']
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = [224, 224]
input_shape = (224, 224, 3)
LOSS = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)

METRICS = ['accuracy']

Epochs = 20
Early_Stop = 15
OPTIMIZER = tensorflow.keras.optimizers.Adam(learning_rate=1e-2, decay=1e-5)

Fine_Tune_Epochs = 20
Fine_Tune_Early_Stop = 15
Fine_Tune_OPTIMIZER = tensorflow.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6)
Fine_Tune_filepath = "Best-Model-FT.h5"

Callbacks = [
    CosineAnnealingScheduler(Epochs, 1e-3, 1e-6),
    EarlyStopping(monitor='val_accuracy', patience=Early_Stop, mode='auto', min_delta=0.001, verbose=2,
                  restore_best_weights=True)]

FT_Callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=2, mode='min', min_delta=0.0001, cooldown=1,
                      min_lr=1e-6),
    ModelCheckpoint(Fine_Tune_filepath, monitor='val_accuracy', verbose=2, save_best_only=True, save_weights_only=False,
                    mode='max'),
    EarlyStopping(monitor='val_loss', patience=Fine_Tune_Early_Stop, mode='auto', min_delta=0.00001, verbose=2,
                  restore_best_weights=True)]

# load_default_dataset()
filenames = get_filenames()
random.shuffle(filenames)

count_data_dict = characterize_data(data_files=filenames)

train_list_ds, val_list_ds, test_list_ds, train_count, val_count, test_count = prepare_datasets(filenames)

train_ds, val_ds, test_ds = convert_imgs(train_ds=train_list_ds, val_ds=val_list_ds, test_ds=test_list_ds)

train_ds = prepare_batches_for_training(train_ds, strategy)
val_ds = prepare_batches_for_training(val_ds, strategy)
test_ds = prepare_batches_for_training(test_ds, strategy, False)

img_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomContrast(factor=0.20)
], name="Augmentation")

display_example_data(ds=train_ds)

x_test, y_test = dataset_to_numpy_util(test_ds, test_count)

print("Evaluation Dataset:")
print('X shape: ', x_test.shape, ' Y shape: ', y_test.shape)

class_weight = generate_weights(count_data_dict, extra_weight=False, ew_value=1.5)

with strategy.scope():
    model = build_model(OPTIMIZER, LOSS, METRICS, NETWORK)

history = fit_model(Epochs, Callbacks, model, train_ds, val_ds, class_weight=class_weight)

Plot_Learning_Curves(history)

results = evaluate_model(model, test_ds)


def fine_tune(OPTIMIZER, LOSS, METRICS):
    for layer in model.layers[-54:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=Fine_Tune_OPTIMIZER, loss=LOSS, metrics=METRICS)
    return model


with strategy.scope():
    model = fine_tune(Fine_Tune_OPTIMIZER, LOSS, METRICS)

history = fit_model(Fine_Tune_Epochs, FT_Callbacks, model, train_ds, val_ds, class_weight=class_weight)
Plot_Learning_Curves(history)
results2 = evaluate_model(model, test_ds)
preds = model.predict(x_test)
print('Shape of preds: ', preds.shape)


generate_results_examples(preds, x_test, y_test)

categories = ['COVID-19', 'Non-COVID-19']
preds = np.round(preds, 0)
class_metrics = metrics.classification_report(y_test, preds, target_names=categories, zero_division=0)
print(class_metrics)
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1))
generate_confusion_matrixes(matrix)

test_image = x_test[0]
x = np.expand_dims(test_image, axis=0)
x = x / 255

images = np.vstack([x])

classes = model.predict(images, batch_size=BATCH_SIZE)
classes = np.argmax(classes, axis=1)

display_other_metrics(classes, matrix, y_test, preds)
