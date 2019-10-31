import pickle
from matplotlib import pyplot as plt


def save_model(model, history):
    # save the model
    model.save('my_model')
    print("The model was save")
    del model  # delete the previous model

    # save the results
    pickle_out = open("accuracy.pickle", "wb")
    pickle.dump(history.history['accuracy'], pickle_out)
    pickle_out.close()
    pickle_out = open("val_accuracy.pickle", "wb")
    pickle.dump(history.history['val_accuracy'], pickle_out)
    pickle_out.close()
    pickle_out = open("loss.pickle", "wb")
    pickle.dump(history.history['loss'], pickle_out)
    pickle_out.close()
    pickle_out = open("val_loss.pickle", "wb")
    pickle.dump(history.history['val_loss'], pickle_out)
    pickle_out.close()

    # Plot training & validation accuracy values
    plt.figure(0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("accuracy plot")
    plt.show()

    # Plot training & validation loss values
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("loss plot")
    plt.show()

    return None
