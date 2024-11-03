import argparse
import fnmatch
import json
import os
import pathlib
import platform
import shutil
import sys
import time
from typing import Tuple, List
import re

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import gridspec
from matplotlib.widgets import Button
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import common


def progress_bar(percent: float, bar_length: int = 30, suffix: str = '', prefix: str = ''):
    bar = '#' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
    sys.stdout.write(f'\r{prefix}[{bar}] {percent * 100:.2f}%{suffix}')
    sys.stdout.flush()


def main(argv: argparse.Namespace):
    start_time = time.time()

    print("python:", platform.python_version())
    print(f"torch {torch.__version__} CUDA: {torch.cuda.is_available()}")
    print("Data folder: ", args.input_folder)

    image_count = sum(len(fnmatch.filter(files, '*.jpg')) for _, _, files in os.walk(args.input_folder))
    print("Image count: ", image_count)
    print(f"Image resolution: {common.IMG_WIDTH}x{common.IMG_HEIGHT}")

    # Hyperparameters
    batch_size = 32
    epochs = 64
    learning_rate = 0.0001
    print(f"batch size: {batch_size}, epochs: {epochs}, lr: {learning_rate}")

    # Data augmentations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((common.IMG_HEIGHT, common.IMG_WIDTH)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize(*common.NORM_VECS)
        ]),
        'val': transforms.Compose([
            transforms.Resize((common.IMG_HEIGHT, common.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(*common.NORM_VECS)
        ])
    }

    # Load datasets
    train_ds = datasets.ImageFolder(os.path.join(args.input_folder, 'train'), transform=data_transforms['train'])
    val_ds = datasets.ImageFolder(os.path.join(args.input_folder, 'val'), transform=data_transforms['val'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Get class names
    class_names = train_ds.classes
    num_classes = len(class_names)
    print("class_names: ", class_names)

    # Instantiate the model
    model = torch.jit.script(common.MyModel(num_classes))
    print("Model JIT: True")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=learning_rate / 10)

    # Training loop
    device = common.get_device()
    print(f"device type: {device.type}")
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

    train_start_time = time.time()
    for epoch in range(epochs):
        print(f">> Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        train_loader_len = len(train_loader.dataset)
        item_count = round(train_loader_len / batch_size)
        print(f"train item_count: {item_count}")

        for idx, (images, labels) in enumerate(train_loader):
            progress_bar(idx / item_count, suffix=f" {idx}/{item_count}")
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
        progress_bar(1, suffix=f" {item_count}/{item_count}\n")

        train_loss = running_loss / train_loader_len
        train_acc = correct_preds / total_preds

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            val_loader_len = len(val_loader.dataset)
            item_count = round(val_loader_len / batch_size)
            print("val item_count: ", item_count)

            for idx, (images, labels) in enumerate(val_loader):
                progress_bar(idx / item_count, suffix=f" {idx}/{item_count}")

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

            progress_bar(1, suffix=f" {item_count}/{item_count}\n")

        val_loss /= val_loader_len
        val_acc = correct_preds / total_preds

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{argv.output_file}.tmp")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered")
                break

    elapsed_time = time.time() - train_start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training complete in {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    print("Evaluate model")
    eva_start_time = time.time()

    evaluate(model, class_names, os.path.join(data_dir, 'eva'), argv.output_file, args.save_plot, args.show_plot)

    elapsed_time = time.time() - eva_start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Evaluation complete in {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    if argv.save_plot or args.show_plot:
        visualize(history, argv.output_file, args.save_plot, args.show_plot)

    # Load the best model for inference
    model.load_state_dict(torch.load(f"{argv.output_file}.tmp", weights_only=False))
    torch.save(model.state_dict(), argv.output_file)
    print(f"Model saved to \"{argv.output_file}\"")

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Full run time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


def visualize(history, model_save_file: str, save_plot: bool, show_plot: bool):
    print("Visualize")

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

    if save_plot:
        ext_start_idx = model_save_file.rfind('.')
        if ext_start_idx != -1:
            plt.savefig(f"{model_save_file[:ext_start_idx]}_visualize.png")

    if show_plot:
        plt.show(block=True)


def evaluate(model, class_names: List[str], eva_dir: str, model_save_file: str, save_plot: bool, show_plot: bool):
    print("Evaluate")
    files: List[str] = []

    for dir_path, _, filenames in os.walk(eva_dir):
        for filename in filenames:
            files.append(os.path.join(dir_path, filename))

    files_len = len(files)

    total_correct = 0
    history = []
    for file in files:
        expected = "ok" if re.match(".*(ok)_.+", file) else "x"
        try:
            c, p = predict(model, class_names, file)
        except RuntimeError:
            files_len -= 1
            print(f"Skipping bad file: {file}", file=sys.stderr)
            continue

        history.append({
            "file": file,
            "expected": expected,
            "got": {
                "class_name": c,
                "confidence_percentage": p
            }
        })

        if expected == c:
            total_correct += 1

    print(f"Accuracy: {total_correct/files_len:.4f}")

    if show_plot or save_plot:
        out_dir = model_save_file.replace(".pt", "_evaluation")

        images_per_page = 3*3
        total_images = files_len

        total_pages = (len(history) + images_per_page - 1) // images_per_page  # Calculate total pages
        current_page = 0

        # Set up the figure and axis for the pagination
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # 3 rows and 3 columns
        axes = axes.flatten()  # Flatten for easy indexing

        # Create buttons for navigation
        ax_prev = plt.axes([0.1, 0.01, 0.1, 0.05])
        ax_next = plt.axes([0.8, 0.01, 0.1, 0.05])
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')

        # Update function for each page
        def update_page(page):
            fig.suptitle(f"Page {page + 1} of {total_pages} | Correct: {total_correct}/{total_images}", fontsize=14)

            start_idx = page * images_per_page
            end_idx = min(start_idx + images_per_page, len(history))

            for i, ax in enumerate(axes):
                ax.clear()
                if start_idx + i < end_idx:
                    entry = history[start_idx + i]
                    img = Image.open(entry['file'])

                    # Expected and got values
                    exp = entry["expected"]
                    got_class = entry["got"]["class_name"]
                    confidence = entry["got"]["confidence_percentage"]

                    # Display image
                    ax.imshow(img)
                    ax.axis('off')

                    # Determine title color based on correctness
                    title_color = 'red' if exp != got_class else 'black'
                    ax.set_title(f"Expected: {exp} | Got: {got_class} ({confidence:.2f}%)", fontsize=10,
                                 color=title_color)

                    # Display filename below image
                    file_name = os.path.basename(entry['file'])
                    ax.text(0.5, -0.1, file_name, fontsize=8, ha='center', va='top', transform=ax.transAxes)
                else:
                    ax.axis('off')  # Hide extra subplots

            plt.draw()
            # Save the current page to a PNG file
            plt.savefig(f"{out_dir}/page_{page + 1}.png", bbox_inches='tight', dpi=300)

        # Button event functions
        def next_page(event):
            nonlocal current_page
            if current_page < total_pages - 1:
                current_page += 1
                update_page(current_page)

        def prev_page(event):
            nonlocal current_page
            if current_page > 0:
                current_page -= 1
                update_page(current_page)

        # Attach event listeners to buttons
        btn_next.on_clicked(next_page)
        btn_prev.on_clicked(prev_page)

        # Initial display
        if save_plot:
            shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(out_dir, exist_ok=False)

            with open(f"{out_dir}/history.json", 'w') as json_file:
                json.dump(history, json_file)

            for p in range(total_pages):
                progress_bar(p / total_pages, 20, prefix="Saving pages ", suffix=f" ({p}/{total_pages})")
                update_page(p)
            progress_bar(1, prefix="Saving pages ", suffix=f" ({total_pages}/{total_pages})\n")

        if show_plot:
            current_page = 0
            update_page(current_page)
            plt.show(block=True)


def predict(model, class_names, img_path: str) -> Tuple[str, float]:
    image = Image.open(img_path)
    image = transforms.Compose([
        transforms.Resize((common.IMG_HEIGHT, common.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(*common.NORM_VECS)
    ])(image).unsqueeze(0).to(common.get_device())

    model.eval()
    with torch.no_grad():
        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)
        confidence_percentage = confidence.item() * 100

        return class_names[predicted.item()], confidence_percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--save-plot', help="saves visualization of plots",
                        required=False, default=False, action='store_true')
    parser.add_argument('-s', '--show-plot', help="shows plots in window", required=False,
                        default=False, action='store_true')
    parser.add_argument('-i', '--input-folder', help="parent folder of train, val and eva dirs",
                        required=False, default=pathlib.Path(os.path.join(os.getcwd(), "cache/data")))
    parser.add_argument('-o', '--output-file', help="where the pt file should be saved",
                        required=True)

    args = parser.parse_args()

    if not args.output_file.endswith(".pt"):
        print("Invalid output file type", file=sys.stderr)
        exit(1)

    main(args)
