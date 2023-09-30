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


def get_dataloaders_and_classes(batch_size=32):
    # Loading train dataset
    train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)

    # Split dataset into train and test
    train_idx, test_idx = train_test_split(list(range(len(train_dataset))), test_size=0.2,
                                           stratify=train_dataset.targets, random_state=42)

    # Creating PyTorch data samplers
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)

    # Loading validation dataset
    val_set = datasets.ImageFolder(root='data/valid', transform=val_test_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    num_classes = len(train_dataset.classes)

    return train_loader, val_loader, test_loader, num_classes


