import matplotlib.pyplot as plt
import numpy as np
from data_characteristics import CLASSES


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(15):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASSES[np.argmax(label_batch[n])])
        plt.axis("off")

    plt.show()


def display_example_data(ds):
    image_batch, label_batch = next(iter(ds))
    show_batch(image_batch.numpy(), label_batch.numpy())
