import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn import metrics
from data_characteristics import CLASSES
from get_config import config



def Plot_Learning_Curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    sns.set(style="dark")
    plt.rcParams['figure.figsize'] = (14, 5)

    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, linestyle="--", label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, linestyle="--", label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    sns.set(style="dark")
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n Accuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def display_other_metrics(matrix, y_test, preds):

    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix[:].sum() - (FP + FN + TP)
    print(TP)
    print(TN)
    print(FP)
    print(FN)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    FDR = FP / (TP + FP)

    ACC = (TP + TN) / (TP + FP + FN + TN)

    print('Other Metrics:')
    MSE = mean_squared_error(y_test, preds)

    print('MSE:', MSE)
    print('Accuracy:', ACC)
    print('Precision (positive predictive value):', PPV)
    print('Recall (Sensitivity, hit rate, true positive rate):', TPR)
    print('Specificity (true negative rate):', TNR)
    print('Negative Predictive Value:', NPV)
    print('Fall out (false positive rate):', FPR)
    print('False Negative Rate:', FNR)
    print('False discovery rate:', FDR)


def generate_results_examples(preds, x_test, y_test):
    plt.figure(figsize=(12, 12))
    R = np.random.choice(preds.shape[0])

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        R = np.random.choice(preds.shape[0])
        pred = np.argmax(preds[R])
        actual = np.argmax(y_test[R])
        col = 'g'
        if pred != actual:
            col = 'r'
        plt.xlabel('I={} | P={} | L={}'.format(R, pred, actual), color=col)
        plt.imshow(((x_test[R] * 255).astype(np.uint8)), cmap='binary')
    plt.show()


def generate_confusion_matrixes(y_test, preds):
    class_metrics = metrics.classification_report(y_test, preds, target_names=CLASSES, zero_division=0)
    print(class_metrics)
    matrix = metrics.confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1))
    draw_matrixes(matrix=matrix)
    return matrix


def draw_matrixes(matrix):
    plot_confusion_matrix(cm=np.array(matrix),
                          normalize=False,
                          target_names=config.content['CLASSES'],
                          title="Confusion Matrix")

    plot_confusion_matrix(cm=np.array(matrix),
                          normalize=True,
                          target_names=config.content['CLASSES'],
                          title="Confusion Matrix, Normalized")


def predict_single(test_image, model, batch_size):
    x = np.expand_dims(test_image, axis=0)
    x = x / 255
    images = np.vstack([x])
    classes = model.predict(images, batch_size=batch_size)
    classes = np.argmax(classes, axis=1)
    categories = CLASSES
    print('Class:', categories[int(classes)])
