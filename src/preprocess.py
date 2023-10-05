import random

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_dataloaders_and_classes(batch_size=32, num_samples_per_class=500):
    train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)

    # Gathering indices per class
    indices_per_class = {}
    for idx, (_, class_idx) in enumerate(train_dataset):
        if class_idx not in indices_per_class:
            indices_per_class[class_idx] = []
        indices_per_class[class_idx].append(idx)

    # Limiting number of samples per class
    limited_indices = []
    for class_idx, indices in indices_per_class.items():
        limited_indices.extend(random.sample(indices, min(num_samples_per_class, len(indices))))

    # Split the limited indices into train and test indices
    train_idx, val_idx = train_test_split(limited_indices, test_size=0.2,
                                          stratify=[train_dataset[i][1] for i in limited_indices], random_state=42)

    # Creating data loaders
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

    test_set = datasets.ImageFolder(root='data/valid', transform=val_test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    num_classes = len(train_dataset.classes)

    return train_loader, val_loader, test_loader, num_classes
