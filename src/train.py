import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import ResNet9
from plots import plot_loss_and_accuracy, plot_confusion_matrix, compute_metrics, plot_metrics
from preprocess import get_dataloaders_and_classes
from src.utils import print_diseases, print_data_frame
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, dev, epoch):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc=f"Training epoch {epoch + 1}", leave=False):
        images, labels = images.to(dev), labels.to(dev)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(loader.dataset), correct / total


def validate_model(model, loader, criterion, dev, epoch):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Validating epoch {epoch + 1}", leave=False):
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader.dataset), correct / total


def train_model(model_class, num_epochs, num_samples_per_class, batch_size, learning_rate):
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    train_loader, val_loader, test_loader, num_classes = (
        get_dataloaders_and_classes(batch_size, num_samples_per_class))

    print_diseases()
    print_data_frame()

    model = model_class(3, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list = [], [], [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device, epoch)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}")

        if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
            plot_loss_and_accuracy(str(model_class.__name__),
                                   train_loss_list,
                                   val_loss_list,
                                   train_accuracy_list,
                                   val_accuracy_list)

            confusion_matrix = torch.zeros(num_classes, num_classes)
            true_labels = []
            predicted_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                    true_labels.extend(labels.cpu().numpy())
                    predicted_labels.extend(predicted.cpu().numpy())

            precision, recall, f1 = compute_metrics(true_labels, predicted_labels)
            print(f"Epoch {epoch + 1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            plot_metrics(model_class, precision, recall, f1, epoch + 1)
            plot_confusion_matrix(model_class, confusion_matrix, val_loader.dataset.classes)

    __save_model(model, model_class)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def __save_model(model, model_class):
    model_save_path = f"models/{model_class.__name__}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved at {model_save_path}')


MODEL_DICT = {
    'ResNet9': ResNet9
}


def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model', type=str, required=True, help='Model name.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--num_samples_per_class', type=int, default=500, help='Number of samples per class.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')

    args = parser.parse_args()

    if args.model in MODEL_DICT:
        model_class = MODEL_DICT[args.model]
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    train_model(model_class, num_epochs=args.epochs, num_samples_per_class=args.num_samples_per_class,
                batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()
