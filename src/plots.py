import matplotlib.pyplot as plt


def plot_loss_and_accuracy(train_loss, val_loss, val_accuracy):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'g', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

