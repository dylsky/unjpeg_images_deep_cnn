import numpy as np
import cv2
import glob
from PIL import Image
import logging
logging.basicConfig(level=logging.INFO)


def get_dataset(default_fullname, images_location, save_path, compress=False, compression_quality=100,
                path_for_compressed_images="", compressed_fullnames=""):
    try:
        a = np.load(default_fullname)
        logging.info("File %s found, loading", default_fullname)
        return a
    except FileNotFoundError:
        logging.info("Numpy file not found at %s", default_fullname)
        if not compress:
            x_data = []
            files = glob.glob(images_location)
            if len(files) > 0:
                for myFile in files:
                    image = cv2.imread(myFile)[:, :, ::-1]  # BGR to RGB
                    x_data.append(image)
                    logging.info("Packing file to numpy array: %s", myFile.title())
                a = np.array(x_data)
                logging.info('Array size:  %s', str(a.shape))
                np.save(save_path, a)
                return a
            else:
                logging.error("No files in directory")
                return None

        else:
            produce_compressed_images(images_location, path_for_compressed_images, compression_quality)
            x_data = []
            files = glob.glob(compressed_fullnames)
            if len(files) > 0:
                for myFile in files:
                    logging.info("Packing file to numpy array: %s", myFile.title())
                    image = cv2.imread(myFile)[:, :, ::-1]  # BGR to RGB
                    x_data.append(image)
                    logging.info("File %s - packed", myFile.title())
                a = np.array(x_data)
                logging.info('Array size: %s', str(a.shape))
                np.save(save_path, a)
                return a
            else:
                logging.error("No files in directory")
                return None


def produce_compressed_images(input_path, output_path, compression_quality):
    files = glob.glob(input_path)
    if len(files) > 0:
        try:
            internal_iterator = 1
            for file in files:
                logging.info("Processing file: %s", file.title())
                compImg = Image.open(file)
                compImg.save(output_path + str(internal_iterator).zfill(7) + ".jpg", "JPEG", quality=compression_quality)
                logging.info("File %s successfully processed", file.title())
                internal_iterator = internal_iterator + 1
        except IOError:
            print("Failed to jpgize. ")
    else:
        logging.error("No files found in source directory %s!", input_path)

