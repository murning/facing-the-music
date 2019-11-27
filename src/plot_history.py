import matplotlib.pyplot as plt
import deep_learning_models
import sys


def plot_accuracy(model_name):
    plt.figure(figsize=(10,5))
    history = deep_learning_models.load_hist(
        'training_history/HISTORY_{model_name}'.format(model_name=model_name))
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model Accuracy: Raw Audio CNN')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig("training_history/{model_name}_accuracy.png".format(model_name=model_name),
                dpi=300)


def plot_loss(model_name):
    plt.figure(figsize=(10,5))
    history = deep_learning_models.load_hist(
        'training_history/HISTORY_{model_name}'.format(model_name=model_name))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss: Raw Audio CNN')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("training_history/{model_name}_loss.png".format(model_name=model_name),
                dpi=300)


if __name__ == '__main__':
    model_name = sys.argv[1]
    plot_accuracy(model_name)
    plot_loss(model_name)
