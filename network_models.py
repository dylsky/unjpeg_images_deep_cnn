from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.models import Model


def basic_model(height, width, depth):
    # deep CNN
    input_img = Input(shape=(height, width, depth))
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = Activation('relu')(x)

    for i in range(15):
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(3, (3, 3), padding='same')(x)
    output_img = Activation('tanh')(x)

    model = Model(input_img, output_img)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model
