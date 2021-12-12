import os
import random

from batches_display import display_example_data
from import_data import load_default_dataset, get_filenames
import numpy as np

from data_characteristics import characterize_data
from model import generate_weights
from prepare_data import prepare_datasets, convert_imgs, prepare_batches_for_training, dataset_to_numpy_util
from statistics import  display_other_metrics, generate_confusion_matrixes, predict_single
from training_config import train_stage, fine_tuning, model_prediction

SEED = 7
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)

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


load_default_dataset()
filenames = get_filenames()
random.shuffle(filenames)

count_data_dict = characterize_data(data_files=filenames)

train_list_ds, val_list_ds, test_list_ds, train_count, val_count, test_count = prepare_datasets(filenames)
train_ds, val_ds, test_ds = convert_imgs(train_ds=train_list_ds, val_ds=val_list_ds, test_ds=test_list_ds)

train_ds = prepare_batches_for_training(train_ds, strategy)
val_ds = prepare_batches_for_training(val_ds, strategy)
test_ds = prepare_batches_for_training(test_ds, strategy, False)

display_example_data(ds=train_ds)

x_test, y_test = dataset_to_numpy_util(test_ds, test_count)
print("Evaluation Dataset:")
print('X shape: ', x_test.shape, ' Y shape: ', y_test.shape)
class_weight = generate_weights(count_data_dict, extra_weight=False, ew_value=1.5)

model_trained = train_stage(strategy, train_ds, val_ds, test_ds, class_weight)
categories = ['COVID-19', 'Non-COVID-19']
model_fine = fine_tuning(strategy, model_trained, train_ds, val_ds, test_ds, class_weight)
preds = model_prediction(model_fine, x_test, y_test)

matrix = generate_confusion_matrixes(y_test=y_test, preds=preds)

test_image = x_test[0]
predict_single(test_image=test_image, model=model_fine, batch_size=BATCH_SIZE)

display_other_metrics(matrix, y_test, preds)
