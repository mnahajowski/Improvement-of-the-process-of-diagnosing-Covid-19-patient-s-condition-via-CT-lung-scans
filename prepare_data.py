from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from data_characteristics import CLASSES
from get_config import config

IMAGE_SIZE = config.content['IMAGE_SIZE']
input_shape = tuple(config.content['input_shape'])
AUTOTUNE = tf.data.experimental.AUTOTUNE


def prepare_datasets(data):
    train_list_ds, val_list_ds, test_list_ds = split_train_test(filenames=data)
    train_count, val_count, test_count = get_sets_statistics(
        train_list_ds=train_list_ds,
        val_list_ds=val_list_ds,
        test_list_ds=test_list_ds)

    return train_list_ds, val_list_ds, test_list_ds, train_count, val_count, test_count


def split_train_test(filenames):
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.2)
    train_filenames, val_filenames = train_test_split(train_filenames, test_size=0.2)
    return create_tensor_slices(
        train_filenames=train_filenames,
        val_filenames=val_filenames,
        test_filenames=test_filenames)


def create_tensor_slices(train_filenames, val_filenames, test_filenames):
    train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
    val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)
    test_list_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
    return train_list_ds, val_list_ds, test_list_ds


def get_sets_statistics(train_list_ds, val_list_ds, test_list_ds):
    train_img_count = tf.data.experimental.cardinality(train_list_ds).numpy()
    print("Training images count: " + str(train_img_count))

    val_img_count = tf.data.experimental.cardinality(val_list_ds).numpy()
    print("Validating images count: " + str(val_img_count))

    test_img_count = tf.data.experimental.cardinality(test_list_ds).numpy()
    print("Testing images count: " + str(test_img_count))

    return train_img_count, val_img_count, test_img_count


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return int(parts[-2] == CLASSES)


def decode_img(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, IMAGE_SIZE)


def convert_imgs(train_ds, val_ds, test_ds, mode=AUTOTUNE):
    return train_ds.map(process_path, num_parallel_calls=mode), \
           val_ds.map(process_path, num_parallel_calls=mode), \
           test_ds.map(process_path, num_parallel_calls=mode)


def prepare_batches_for_training(ds, strategy, cache=True):
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)

    if cache:
        ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def dataset_to_numpy_util(dataset, N):
    dataset = dataset.unbatch().batch(N)
    for images, labels in dataset:
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        break
    return numpy_images, numpy_labels
