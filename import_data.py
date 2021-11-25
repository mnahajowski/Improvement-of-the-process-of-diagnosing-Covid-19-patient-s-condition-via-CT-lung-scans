from zipfile import ZipFile
import tensorflow as tf
from kaggle import KaggleApi
from get_config import config

DEFAULT_COVID_DATASET_URL = 'mehradaria/covid19-lung-ct-scans'
DEFAULT_COVID_DATA_PATH = 'COVID-19_Lung_CT_Scans/COVID-19/*'
DEFAULT_NON_COVID_DATA_PATH = 'COVID-19_Lung_CT_Scans/Non-COVID-19/*'
DEFAULT_INVALID_FILES = [r'COVID-19_Lung_CT_Scans\COVID-19\desktop.ini']


def authorize_kaggle():
    api = KaggleApi()
    api.authenticate()
    return api


def import_default_kaggle_dataset():
    API = authorize_kaggle()
    API.dataset_download_files(DEFAULT_COVID_DATASET_URL)


def extract_dataset_zip_file():
    zf = ZipFile(DEFAULT_COVID_DATASET_URL.split('/')[-1] + '.zip')
    zf.extractall()
    zf.close()


def load_default_dataset():
    import_default_kaggle_dataset()
    extract_dataset_zip_file()
    get_filenames()


def get_filenames():
    if config.content['custom_covid_data_path'] == '':
        filenames = tf.io.gfile.glob(str(DEFAULT_COVID_DATA_PATH))
        n = int(round(0.3 * len(filenames)))
        # filenames = filenames[:len(filenames) - n]
        # filenames.extend(tf.io.gfile.glob(str(DEFAULT_NON_COVID_DATA_PATH)))
        m = int(round(0.3 * len(tf.io.gfile.glob(str(DEFAULT_NON_COVID_DATA_PATH)))))
        data = tf.io.gfile.glob(str(DEFAULT_NON_COVID_DATA_PATH))
        # data = data[:len(data) - m]
        filenames.extend(data)
        # filenames = remove_invalid_files(filenames=filenames)
        # filenames = filenames[:len(filenames) - m]
    else:
        filenames = tf.io.gfile.glob(str(config.content['custom_covid_data_path'] + '*'))
        data = tf.io.gfile.glob(str(config.content['custom_non_covid_data_path'] + '*'))
        filenames.extend(data)
    return filenames


def remove_invalid_files(filenames):
    filenames = set(filenames) - set(DEFAULT_INVALID_FILES)
    return list(filenames)
