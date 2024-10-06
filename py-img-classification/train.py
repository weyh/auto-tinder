import argparse
import fnmatch
import os
import pathlib
import platform
import sys
import time
from functools import lru_cache

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


def mprint(text: str):
    print(">>", text, sep=' ')


def progress_bar(percent: float, bar_length: int = 30, suffix: str = '', prefix: str = ''):
    bar = '#' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
    sys.stdout.write(f'\r{prefix}[{bar}] {percent * 100:.2f}%{suffix}')
    sys.stdout.flush()


@lru_cache(maxsize=1)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.rocm.is_available():
        return torch.device("rocm")
    return torch.device("cpu")


class CNNModel(nn.Module):
    def __init__(self, num_classes, img_width: int, img_height: int):
        super(CNNModel, self).__init__()
        self.data_augmentation = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 input channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (img_height // 16) * (img_width // 16), 256)  # Adjust the size
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.data_augmentation(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def main(argv: argparse.Namespace):
    mprint(f"python: {platform.python_version()}")
    mprint(f"torch {torch.__version__} CUDA: {torch.cuda.is_available()}")

    data_dir = pathlib.Path(os.path.join(os.getcwd(), "cache/data"))
    image_count = sum(len(fnmatch.filter(files, '*.jpg')) for _, _, files in os.walk(data_dir))
    mprint(f"Image count: {image_count}")

    # Hyperparameters
    img_height = 180
    img_width = 180
    batch_size = 32
    epochs = 2 # 50
    learning_rate = 0.001

    norm_vecs = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Data augmentations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((img_height, img_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize(*norm_vecs)  # Normalization
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(*norm_vecs)
        ])
    }

    # Load datasets
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms['val'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Get class names
    class_names = train_ds.classes
    num_classes = len(class_names)
    mprint(f'class_names: {class_names}')

    # Instantiate the model
    model = torch.jit.script(CNNModel(num_classes, img_width=img_width, img_height=img_height))
    mprint("Model JIT: True")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=0.0001)

    # Training loop
    device = get_device()
    mprint(f"device type: {device.type}")
    model.to(device)

    best_val_loss = float('inf')
    early_stopping_patience = 5
    epochs_no_improve = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        mprint(f"Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        train_loader_len = len(train_loader.dataset)
        item_count = round(train_loader_len / batch_size)
        mprint(f"train item_count: {item_count}")

        for idx, (images, labels) in enumerate(train_loader):
            progress_bar((idx + 1) / item_count, suffix=f" {idx + 1}/{item_count}")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
        print()

        train_loss = running_loss / train_loader_len
        train_acc = correct_preds / total_preds

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        mprint(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            val_loader_len = len(val_loader.dataset)
            item_count = round(val_loader_len / batch_size)
            mprint(f"val item_count: {item_count}")

            for idx, (images, labels) in enumerate(val_loader):
                progress_bar((idx + 1) / item_count, suffix=f" {idx + 1}/{item_count}")

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)
            print()

        val_loss /= val_loader_len
        val_acc = correct_preds / total_preds

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        mprint(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{argv.out_file}.tmp")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                mprint("Early stopping triggered")
                break

    mprint("Training complete!")

    if argv.show:
        visualize(history, argv.out_file)
    if argv.ref_file is not None:
        predict(model, class_names, norm_vecs, img_height, img_width, argv.ref_file)

    # Load the best model for inference
    model.load_state_dict(torch.load(f"{argv.out_file}.tmp"))
    torch.save(model.state_dict(), argv.out_file)
    mprint(f"Model saved to \"{argv.out_file}\"")


def visualize(history, model_save_file: str):
    mprint("Visualize")

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'r', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    ext_start_idx = model_save_file.rfind('.')
    if ext_start_idx != -1:
        plt.savefig(f"{model_save_file[:ext_start_idx]}.png")

    #plt.show(block=True)


def predict(model, class_names, norm_vecs: tuple, img_height: int, img_width: int, img_path: str):
    image = Image.open(img_path)
    image = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(*norm_vecs)
    ])(image).unsqueeze(0).to(get_device())

    model.eval()
    with torch.no_grad():
        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)
        confidence_percentage = confidence.item() * 100
        mprint(f'Predicted class: {class_names[predicted.item()]} with confidence: {confidence_percentage:.2f}%')
    time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show', help="shows training history on graph", required=False,
                        default=False, action='store_true')
    parser.add_argument('-r', '--ref-file', help="test out trained model on reference img",
                        required=False, default=None)
    parser.add_argument('-o', '--out-file', help="where the pth file should be saved",
                        required=True)

    args = parser.parse_args()

    if not args.out_file.endswith(".pt"):
        print("Invalid output file type", file=sys.stderr)
        exit(1)

    main(args)
