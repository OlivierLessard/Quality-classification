from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2.
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylim([-0.5, 1.5]) # To fix the size limitation
    fig.tight_layout()

    return ax


def test_model(headline, oos_y, oos_pred, CATEGORIES):  # oos_y shape : [n_features]
    # acc, prec, recall, f1
    print_metrics(headline, oos_y, oos_pred)

    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plot_confusion_matrix(oos_y, oos_pred, classes=CATEGORIES,
                          normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
    plt.savefig("confusion matrix")


def print_metrics(headline, y_test, pred):
    print(headline)
    print("balanced accuracy: {}".format(balanced_accuracy_score(y_test, pred)))
    print("precision: {}".format(precision_score(y_test, pred)))
    print("recall: {}".format(recall_score(y_test, pred)))
    print("f1: {}".format(f1_score(y_test, pred)))

