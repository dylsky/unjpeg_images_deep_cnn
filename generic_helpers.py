from enum import Enum


# this is the class for storing relative paths
class Paths(Enum):
    source_images = "./train/*.png"
    source_images_package = "./train.npy"
    source_images_package_name = "train"
    compressed_images = "./train_comp/*.jpg"
    compressed_images_save_path = "./train_comp/"
    compressed_images_package = "./train_comp.npy"
    compressed_images_package_name = "train_comp"

    test_source_images = "./test/*.png"
    test_source_images_package = "./test.npy"
    test_source_images_package_name = "test"
    test_compressed_images = "./test_comp/*.jpg"
    test_compressed_images_save_path = "./test_comp/"
    test_compressed_images_package = "./test_comp.npy"
    test_compressed_images_package_name = "test_comp"


# display first (default 10) elements of two numpy image arrays in a mathplotlib way
def demo_two_rows(orig, noise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(orig[i].reshape(64, 64, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(noise[i].reshape(64, 64, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# display first (default 10) elements of three numpy image arrays in a mathplotlib way
def demo_three_rows(orig, noise, denoise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 6))

    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(orig[i].reshape(64, 64, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noise[i].reshape(64, 64, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display denoised image
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(denoise[i].reshape(64, 64, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def normallize_import_data(normalization_array):
    normalization_array = normalization_array.astype('float32')
    normalization_array /= 255
    return normalization_array


def run_tensorboard():
    def launch_tb():
        import os
        os.system('tensorboard --logdir=./tb --port 6789')
        return

    import threading
    import webbrowser
    t = threading.Thread(target=launch_tb, args=([]))
    t.start()

    webbrowser.open_new("http:\\\\desktop-j3qujuk:6789")

