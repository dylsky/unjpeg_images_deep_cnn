import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import logging
import network_models, dataset_helpers
from generic_helpers import Paths, demo_three_rows, normallize_import_data, run_tensorboard

logging.basicConfig(level=logging.INFO)


def main():
    os.environ["KERAS_BACKEND"] = "tensorflow"  # to not use Theano accidentaly
    batch_size = 32
    epochs = 32
    model_checkpoint = 'DeepCNN.hdf5'

    #run tensorboard. You may have to refresh tb browser window at it isn't ready instantly
    run_tensorboard()

    x_train = dataset_helpers.get_dataset(default_fullname=Paths.source_images_package.value,
                                          images_location=Paths.source_images.value,
                                          save_path=Paths.source_images_package_name.value)

    x_test = dataset_helpers.get_dataset(default_fullname=Paths.test_source_images_package.value,
                                         images_location=Paths.test_source_images.value,
                                         save_path=Paths.test_source_images_package_name.value)

    x_train_noisy = dataset_helpers.get_dataset(default_fullname=Paths.compressed_images_package.value,
                                                images_location=Paths.source_images.value,
                                                save_path=Paths.compressed_images_package_name.value,
                                                compress=True,
                                                compression_quality=50,
                                                path_for_compressed_images=Paths.compressed_images_save_path.value,
                                                compressed_fullnames=Paths.compressed_images.value)

    x_test_noisy = dataset_helpers.get_dataset(default_fullname=Paths.test_compressed_images_package.value,
                                               images_location=Paths.test_source_images.value,
                                               save_path=Paths.test_compressed_images_package_name.value,
                                               compress=True,
                                               compression_quality=50,
                                               path_for_compressed_images=Paths.test_compressed_images_save_path.value,
                                               compressed_fullnames=Paths.test_compressed_images.value)

    # normalize data
    x_train = normallize_import_data(x_train)
    x_test = normallize_import_data(x_test)
    x_train_noisy = normallize_import_data(x_train_noisy)
    x_test_noisy = normallize_import_data(x_test_noisy)

    logging.info('Train data shape: %s', str(x_train.shape))
    logging.info('%s train samples', str(x_train.shape[0]))
    logging.info('%s test samples', str(x_test.shape[0]))

    model = network_models.basic_model(64, 64, 3)

    # load pretrained weights
    try:
        model.load_weights(model_checkpoint)
    except Exception:  # what kind of these can be there? idk. Anyway, one or two more training iterations won't harm
        print("Old weights not found, produce new ones")

    model.fit(x_train_noisy, x_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,  # verbose=1 is too boring :)
              validation_data=(x_test_noisy, x_test),
              callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto'),
                         ModelCheckpoint(filepath=model_checkpoint, monitor='val_loss', verbose=1,
                                         save_best_only=True, mode='auto'),
                         TensorBoard(log_dir='./tb', histogram_freq=0,
                                     write_graph=False)],
              shuffle=True)

    score = model.evaluate(x_test_noisy, x_test, verbose=1)
    logging.info("Model evaluation score: %s", str(score))

    images_processed = model.predict(x_test_noisy)
    images_processed = np.abs(images_processed)

    demo_three_rows(x_test, x_test_noisy, images_processed)


main()
