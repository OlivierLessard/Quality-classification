import pickle


def save_model(model, model_name):
    model.save(model_name)
    print("The model was save")
    del model  # delete the previous model

    return None


def save_test_and_pred(oos_y, oos_pred):
    pickle_out = open("oos_y.pickle", "wb")
    pickle.dump(oos_y, pickle_out)
    pickle_out.close()

    pickle_out = open("oos_pred.pickle", "wb")
    pickle.dump(oos_pred, pickle_out)
    pickle_out.close()

