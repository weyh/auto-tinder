import argparse
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras
from keras import layers
from keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def mprint(text):
    print(">> ", end='')
    print(text)

def get_optimizer():
    if "2.10" in tf.__version__:
        return keras.optimizers.Adam(learning_rate=0.001)

    return tf.keras.optimizers.AdamW(learning_rate=0.0005)


def main(args: argparse.Namespace):
    mprint(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    tf.function(jit_compile=True)
    mprint("JIT ON")

    #tf.debugging.set_log_device_placement(True)
    data_dir = pathlib.Path(os.path.join(os.getcwd(), "cache/data"))
    image_count = len(list(data_dir.glob('*/*.jpg')))
    mprint(image_count)

    batch_size = 32
    img_height = 180
    img_width = 180

    mprint("train_ds")

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    mprint("val_ds")
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names  # type: ignore
    print(f"class_names: {class_names}")

    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1)
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),  # Increased dropout rate
        layers.Conv2D(128, 3, padding='same', activation='relu'),  # Additional layer
        layers.MaxPooling2D(),
        layers.Dropout(0.3),  # Additional dropout
        layers.Flatten(),
        layers.Dense(256, activation='relu'),  # Increased dense layer size
        layers.Dropout(0.5),  # Increased dropout rate
        layers.Dense(num_classes, name="outputs")
    ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    model.compile(optimizer=get_optimizer(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    mprint("Fit")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        callbacks=[early_stopping, reduce_lr]
    )

    if args.show:
        visualize(history, args.out_file)
    if args.ref_file is not None:
        predict(model, class_names, img_height, img_width, args.ref_file)

    mprint(f"Saving model: {args.out_file}")
    model.save(args.out_file)


def visualize(history, model_save_file: str):
    mprint("Visualize")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    ext_start_idx = model_save_file.rfind('.')
    if ext_start_idx != -1:
        plt.savefig(f"{model_save_file[:ext_start_idx]}.png")

    plt.show(block=True)


def predict(model, class_names, img_height: int, img_width: int, ref_path: str):
    mprint("Predict on new data")

    img = tf.keras.utils.load_img(
        ref_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show', help="shows training history on graph", required=False, default=False, action='store_true')
    parser.add_argument('-r', '--ref-file', help="test out trained model on reference img", required=False, default=None)
    parser.add_argument('-o', '--out-file', help="where the tflite file should be saved", required=True)

    args = parser.parse_args()

    main(args)
