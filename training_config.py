

import numpy as np
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow.keras
from model import build_model, fit_model, evaluate_model
from statistics import Plot_Learning_Curves, generate_results_examples
from tensorflow.keras.applications import NASNetLarge
import tensorflow as tf

NETWORK = NASNetLarge
METRICS = ['accuracy']
LOSS = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)


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
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + np.math.cos(np.math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


Epochs = 5
Early_Stop = 5
OPTIMIZER = tensorflow.keras.optimizers.Adam(learning_rate=1e-2, decay=1e-5)

Fine_Tune_Epochs = 5
Fine_Tune_Early_Stop = 5
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


def train_stage(strategy, train_ds, val_ds, test_ds, class_weight):
    with strategy.scope():
        model = build_model(OPTIMIZER, LOSS, METRICS, NETWORK)
    history = fit_model(Epochs, Callbacks, model, train_ds, val_ds, class_weight=class_weight)
    Plot_Learning_Curves(history)

    evaluate_model(model, test_ds)
    return model


def fine_tuning(strategy, model_trained, train_ds, val_ds, test_ds, class_weight):
    with strategy.scope():
        model = fine_tune(model_trained, Fine_Tune_OPTIMIZER, LOSS, METRICS)

    history = fit_model(Fine_Tune_Epochs, FT_Callbacks, model, train_ds, val_ds, class_weight=class_weight)
    Plot_Learning_Curves(history)
    evaluate_model(model, test_ds)
    return model


def fine_tune(model, OPTIMIZER, LOSS, METRICS):
    for layer in model.layers[-54:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    return model


def model_prediction(model, x_test, y_test):
    preds = model.predict(x_test)
    print('Shape of preds: ', preds.shape)
    generate_results_examples(preds, x_test, y_test)
    preds = np.round(preds, 0)
    return preds
