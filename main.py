import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from matplotlib import pyplot as plt
import csv
import pathlib
import PIL
from PIL import Image
import dataset_builder


if __name__ == '__main__':
    train_ds, val_ds, id_to_name_map = dataset_builder.get_dataset()

    print(train_ds)
    print(len(train_ds.class_names))

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"), cmap='gray', vmin=0, vmax=255)
    #         plt.title(id_to_name_map[train_ds.class_names[labels[i]]])
    #         plt.axis("off")
    #
    #     plt.show()

    # Model / data parameters
    num_classes = len(train_ds.class_names)
    # input_shape = (178, 128, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),

        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, 3, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=train_ds,
        epochs=5
    )

    test_loss, test_acc = model.evaluate(train_ds, verbose=2)

    if not os.path.exists('models/'):
        os.makedirs('models/')

    model.save('models/first-test.h5')

    print('\nTest accuracy:', test_acc)
