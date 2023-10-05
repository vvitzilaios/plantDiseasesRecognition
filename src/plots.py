import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def save_plot(plot, model_class, plot_name):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    filename = f"{str(model_class.__name__)}_{plot_name}.png"
    path = os.path.join('plots', filename)
    plot.savefig(path)
    print(f"Plot saved at {path}")


def plot_loss_and_accuracy(model_name, train_loss, val_loss, train_accuracy, val_accuracy):
    epochs = range(1, len(train_loss) + 1)

    # Plotting Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'g', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    save_plot(plt, model_name, 'loss')
    plt.show()

    # Plotting Training and Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracy, 'g', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    save_plot(plt, model_name, 'accuracy')
    plt.show()


def plot_confusion_matrix(model_class, confusion_matrix, classes, epoch):
    plt.figure(figsize=(15, 15))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    save_plot(plt, model_class, f'confusion_matrix_epoch_{epoch}')
    plt.show()


def plot_metrics(model_class, precision, recall, f1_scr, epoch):
    plt.figure(figsize=(10, 5))
    metrics = [precision, recall, f1_scr]
    names = ['Precision', 'Recall', 'F1 Score']

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        plt.bar(names[i], metric)
        plt.title(f'{names[i]} at Epoch {epoch}')
        plt.ylim(0, 1)

    plt.tight_layout()
    save_plot(plt, model_class, f'metrics_epoch_{epoch}.png')
    plt.show()


def compute_metrics(true_labels, predicted_labels, average_method='macro'):
    precision = precision_score(true_labels, predicted_labels, average=average_method)
    recall = recall_score(true_labels, predicted_labels, average=average_method)
    f1 = f1_score(true_labels, predicted_labels, average=average_method)

    return precision, recall, f1
