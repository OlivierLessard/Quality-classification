from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
from save_model import save_model, save_test_and_pred


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
    # ax.set(xticks=np.arange(cm.shape[1]),
    #        yticks=np.arange(cm.shape[0]),
    #        # ... and label them with the respective list entries
    #        xticklabels=classes, yticklabels=classes,
    #        title=title,
    #        ylabel='Vrai classe',
    #        xlabel='Classe predite')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylim([1.5, -0.5])  # To fix the size limitation
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
    plt.savefig("confusion matrix")
    plt.show()


def print_metrics(headline, y_test, pred):
    f = open("metric results.txt", "a+")
    f.write("\n")
    f.write(headline)
    f.write("\naccuracy: {} \n".format(accuracy_score(y_test, pred)))
    f.write("balanced accuracy: {} \n".format(balanced_accuracy_score(y_test, pred)))
    f.write("precision: {} \n".format(precision_score(y_test, pred)))
    f.write("recall: {} \n".format(recall_score(y_test, pred)))
    f.write("f1: {} \n".format(f1_score(y_test, pred)))
    f.close()
    print(headline)
    print("accuracy: {}".format(accuracy_score(y_test, pred)))
    print("balanced accuracy: {}".format(balanced_accuracy_score(y_test, pred)))
    print("precision: {}".format(precision_score(y_test, pred)))
    print("recall: {}".format(recall_score(y_test, pred)))
    print("f1: {}".format(f1_score(y_test, pred)))


def print_metrics2(headline, y_test, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(y_test, pred)))
    print("balanced accuracy: {}".format(balanced_accuracy_score(y_test, pred)))
    print("precision: {}".format(precision_score(y_test, pred)))
    print("recall: {}".format(recall_score(y_test, pred)))
    print("f1: {}".format(f1_score(y_test, pred)))


if __name__ == "__main__":
    with open('y_test_1.pickle', 'rb') as data:
        y_test_1 = pickle.load(data)
        data.close()
    with open('y_pred_1.pickle', 'rb') as data:
        y_pred_1 = pickle.load(data)
        data.close()
    with open('y_test_2.pickle', 'rb') as data:
        y_test_2 = pickle.load(data)
        data.close()
    with open('y_pred_2.pickle', 'rb') as data:
        y_pred_2 = pickle.load(data)
        data.close()
    with open('y_test_3.pickle', 'rb') as data:
        y_test_3 = pickle.load(data)
        data.close()
    with open('y_pred_3.pickle', 'rb') as data:
        y_pred_3 = pickle.load(data)
        data.close()
    with open('y_test_4.pickle', 'rb') as data:
        y_test_4 = pickle.load(data)
        data.close()
    with open('y_pred_4.pickle', 'rb') as data:
        y_pred_4 = pickle.load(data)
        data.close()
    # with open('y_test_5.pickle', 'rb') as data:
    #     y_test_5 = pickle.load(data)
    #     data.close()
    # with open('y_pred_5.pickle', 'rb') as data:
    #     y_pred_5 = pickle.load(data)
    #     data.close()

    oos_y = []
    oos_pred = []

    oos_y.append(y_test_1)
    oos_y.append(y_test_2)
    oos_y.append(y_test_3)
    oos_y.append(y_test_4)
    # oos_y.append(y_test_5)

    oos_pred.append(y_pred_1)
    oos_pred.append(y_pred_2)
    oos_pred.append(y_pred_3)
    oos_pred.append(y_pred_4)
    # oos_pred.append(y_pred_5)

    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    save_test_and_pred(oos_y, oos_pred)
    test_model("\n Final results : ", np.argmax(oos_y, axis=1), np.argmax(oos_pred, axis=1), ["Bad", "Workable"])

    with open('oos_pred.pickle', 'rb') as data:
        oos_pred = pickle.load(data)
        data.close()
    with open('oos_y.pickle', 'rb') as data:
        oos_y = pickle.load(data)
        data.close()

    plot_confusion_matrix(np.argmax(oos_y, axis=1), np.argmax(oos_pred, axis=1), ["Mauvaise", "Utilisable"], True, title="Matrice de confusion normalisee")
    plt.savefig("confusion matrix")
    plt.show()

