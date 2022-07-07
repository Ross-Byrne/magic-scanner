import tensorflow as tf


def main():
    # Test tensorflow-gpu works
    print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    main()
