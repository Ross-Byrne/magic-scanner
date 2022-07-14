import os
import csv
from PIL import Image
from tqdm import tqdm

idToNameMap = {}
idToIndexMap = {}
indexToIdMap = {}


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
            idToNameMap[line[0]] = line[1]
            idToIndexMap[line[0]] = i
            indexToIdMap[i] = line[0]
            i += 1

    # create dataset label csv
    with open("data/mtg-ds/labels.csv", 'w', newline="") as file:
        csvwriter = csv.writer(file, delimiter=",")
        # Add csv headers
        csvwriter.writerow(['index', 'uuid'])

        for uuid, index in idToIndexMap.items():
            csvwriter.writerow([index, uuid])

    # create image name map csv
    with open("data/mtg-ds/image-name-map.csv", 'w', newline="") as file:
        csvwriter = csv.writer(file, delimiter=",")
        # Add csv headers
        csvwriter.writerow(['uuid', 'name'])

        for uuid, name in idToNameMap.items():
            csvwriter.writerow([uuid, name])


def _process_images():
    print("Processing images, please wait...")
    target_width = 128  # should resize images to 128 x 178

    # iterate through cards in data directory
    with os.scandir("data/magic-the-gathering-cards") as dirs:
        for entry in dirs:
            # remove file if new file
            if entry.is_file() and entry.name.split('.')[-1] == 'jpg':
                try:
                    name = entry.name.split('.')[0]
                    image = Image.open(f"data/magic-the-gathering-cards/{entry.name}")
                    image = image.convert("L")  # convert to grayscale
                    image_width, image_height = image.size

                    # calculate resize that preserves aspect ratio
                    ratio = target_width / float(image_width)
                    new_dimensions = (target_width, int(image_height * ratio))
                    image = image.resize(new_dimensions)

                    # save processed image
                    image_label = idToIndexMap[name]
                    image.save(f'data/mtg-ds/images/{image_label}.jpg')
                except OSError as e:  # if failed, report it back to the user
                    print("Error: %s - %s." % (entry.name, e.strerror))

# load files into dataset object
# preprocess dataset


def get_dataset():
    _ensure_folder_structure()
    _create_dataset_labels_csv()
    _process_images()
    return 'dataset here'


if __name__ == '__main__':
    get_dataset()
