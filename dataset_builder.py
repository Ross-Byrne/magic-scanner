import os
import csv
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

id_to_name_map = {}
id_to_index_map = {}
index_to_id_map = {}
image_height = 178
image_width = 128


# make new folder to put processed dataset in
def _ensure_folder_structure():
    if not os.path.exists('data/'):
        os.makedirs('data/')

    if not os.path.exists('data/mtg-ds'):
        os.makedirs('data/mtg-ds')

    if not os.path.exists('data/mtg-ds/images'):
        os.makedirs('data/mtg-ds/images')


def _create_dataset_labels_csv():
    # opening the CSV file
    with open('data/magic-the-gathering-card-labels.csv', mode='r') as file:
        # reading the CSV file
        csv_file = csv.reader(file)
        next(csv_file, None)  # skip the headers
        i = 0

        # create id/name mappings
        for line in csv_file:
            id_to_name_map[line[0]] = line[1]
            id_to_index_map[line[0]] = i
            index_to_id_map[i] = line[0]
            i += 1

    # create dataset label csv
    with open("data/mtg-ds/labels.csv", 'w', newline="") as file:
        csvwriter = csv.writer(file, delimiter=",")
        # Add csv headers
        csvwriter.writerow(['index', 'uuid'])

        for uuid, index in id_to_index_map.items():
            csvwriter.writerow([index, uuid])

    # create image name map csv
    with open("data/mtg-ds/image-name-map.csv", 'w', newline="") as file:
        csvwriter = csv.writer(file, delimiter=",")
        # Add csv headers
        csvwriter.writerow(['uuid', 'name'])

        for uuid, name in id_to_name_map.items():
            csvwriter.writerow([uuid, name])


def _process_images():
    print("Processing images, please wait...")

    # iterate through cards in data directory
    with os.scandir("data/magic-the-gathering-cards") as dirs:
        for entry in dirs:
            # remove file if new file
            if entry.is_file() and entry.name.split('.')[-1] == 'jpg':
                try:
                    name = entry.name.split('.')[0]
                    image = Image.open(f"data/magic-the-gathering-cards/{entry.name}")
                    image = image.convert("L")  # convert to grayscale

                    new_dimensions = (image_width, image_height)
                    image = image.resize(new_dimensions)

                    # save processed image
                    # image_label = idToIndexMap[name]

                    # create folder for class name
                    if not os.path.exists(f'data/mtg-ds/images/{name}'):
                        os.makedirs(f'data/mtg-ds/images/{name}')

                    image.save(f'data/mtg-ds/images/{name}/0.jpg')  # save image as index 0 for now
                except OSError as e:  # if failed, report it back to the user
                    print("Error: %s - %s." % (entry.name, e.strerror))


def _initialise_builder():
    _ensure_folder_structure()
    _create_dataset_labels_csv()


def process_dataset():
    _initialise_builder()
    _process_images()


def get_dataset():
    _initialise_builder()

    train_ds = keras.utils.image_dataset_from_directory(
        directory='data/mtg-ds/images/',
        # directory='data/test-data/',
        batch_size=64,
        seed=123,
        image_size=(image_height, image_width),
        color_mode='grayscale')

    # val_ds = keras.utils.image_dataset_from_directory(
    #     directory='data/mtg-ds/images/',
    #     # directory='data/test-data/',
    #     validation_split=0.2,
    #     subset="validation",
    #     batch_size=132,
    #     seed=123,
    #     image_size=(image_height, image_width),
    #     color_mode='grayscale')

    return train_ds, None, id_to_name_map


if __name__ == '__main__':
    get_dataset()
