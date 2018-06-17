import cv2
import keras
import numpy as np
from image_slicer import *

from django_ui.neural_network_backend.generic_helpers import normallize_import_data, denormallize_import_data
from django_ui.neural_network_backend.network_models import basic_model


def predict(sizex, sizey, number_of_layers, model_path, image_path):
    model = basic_model(sizex, sizey, number_of_layers)
    model.load_weights(model_path)

    x_data = []
    image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
    image = normallize_import_data(image)
    x_data.append(image)
    source = np.array(x_data)
    target = model.predict(source)[0]
    target = denormallize_import_data(target)
    cv2.imwrite("image_after.jpg", target[:, :, ::-1])  # BGR to RGB
    keras.backend.clear_session()
