import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import ResNet9
from plots import plot_loss_and_accuracy
from preprocess import get_dataloaders_and_classes
from src.utils import print_diseases, print_data_frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model_name, num_epochs, batch_size, learning_rate):
    train_loader, val_loader, test_loader, num_classes = get_dataloaders_and_classes(batch_size=batch_size)

    print_diseases()
    print_data_frame()

    model = model_name(3, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list, val_loss_list, val_accuracy_list = [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            print(f"Batch {len(train_loader)} of {len(train_loader)}")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_list.append(epoch_loss)

        # Validate the model
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss_list.append(val_loss / len(val_loader.dataset))
        val_accuracy_list.append(correct / total)

        print(
            f"Epoch {epoch + 1}, Train Loss: {epoch_loss}, Val Loss: {val_loss / len(val_loader.dataset)}, Val "
            f"Accuracy: {correct / total}")

    plot_loss_and_accuracy(train_loss_list, val_loss_list, val_accuracy_list)

    # Evaluate on test data
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total}")


MODEL_DICT = {
    'ResNet9': ResNet9
}


def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model', type=str, required=True, help='Model name.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')

    args = parser.parse_args()

    if args.model in MODEL_DICT:
        model = MODEL_DICT[args.model]
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    train_model(model, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()
