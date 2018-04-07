# coding=utf-8

import os
import time
import numpy as np

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, GRU, LSTM, Dense, Dropout, Permute, Reshape, Flatten, \
    ELU, GlobalMaxPool2D
from keras.layers.merge import concatenate, add, dot
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint

from keras import models

from utils.load_dataset import load_dataset
from utils.sliding_window import sliding_window, opp_sliding_window
from config import *


def cnn_block(input_tensor, feature_maps=NUM_FILTERS):
    l1 = ELU()(
        Conv2D(filters=feature_maps, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(input_tensor))

    l21 = ELU()(
        Conv2D(filters=feature_maps, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(input_tensor))
    l22 = ELU()(
        Conv2D(filters=feature_maps, kernel_size=(1, 3), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(l21))

    l31 = ELU()(
        Conv2D(filters=feature_maps, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(input_tensor))
    l32 = ELU()(
        Conv2D(filters=feature_maps, kernel_size=(1, 5), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(l31))

    l41 = MaxPooling2D(pool_size=(1, 5), strides=(1, 1), padding='same',
                       data_format='channels_last')(input_tensor)
    l42 = ELU()(
        Conv2D(filters=feature_maps, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(l41))
    layer = ELU()(concatenate([l1, l22, l32, l42]))

    return layer


def prepare_data():
    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')

    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NUM_SENSOR_CHANNELS))
    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NUM_SENSOR_CHANNELS))

    return X_train, X_test, y_train, y_test


def train():
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # Networks
    inputs = Input(shape=(1, SLIDING_WINDOW_LENGTH, NUM_SENSOR_CHANNELS))
    conv1 = cnn_block(inputs)
    conv2 = cnn_block(conv1)
    conv3 = cnn_block(conv2)
    maxpool1 = MaxPooling2D(pool_size=(1, SLIDING_WINDOW_LENGTH / 2), strides=(1, 1), padding='same',
                            data_format='channels_last')(conv3)
    conv4 = cnn_block(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(1, SLIDING_WINDOW_LENGTH / 2), strides=(1, 1), padding='valid',
                            data_format='channels_last')(conv4)
    # conv5 = ELU()(
    #     Conv2D(filters=NUM_FILTERS*4, kernel_size=(1, 12), strides=(1, 1), padding='same',
    #            kernel_initializer='random_normal', data_format='channels_last')(maxpool2))
    reshape1 = Reshape((1, (SLIDING_WINDOW_LENGTH - SLIDING_WINDOW_LENGTH / 2 + 1) * NUM_FILTERS * 4))(maxpool2)
    dropout1 = Dropout(DROPOUT_RATE)(reshape1)
    gru1 = GRU(NUM_UNITS_LSTM, return_sequences=True, implementation=2)(dropout1)
    dropout2 = Dropout(DROPOUT_RATE)(gru1)
    gru2 = GRU(NUM_UNITS_LSTM, return_sequences=False, implementation=2)(dropout2)  # implementation=2 for GPU
    dropout3 = Dropout(DROPOUT_RATE)(gru2)
    outputs = Dense(NUM_CLASSES, activation=K.softmax, activity_regularizer=l2())(dropout3)

    model = Model(inputs=inputs, outputs=outputs)

    # Save checkpoints
    timestamp = str(int(time.time()))
    os.mkdir('./runs/%s/' % timestamp)
    checkpoint = ModelCheckpoint('./runs/%s/weights.{epoch:03d}-{val_acc:.4f}.hdf5' % timestamp, monitor='val_acc',
                                 verbose=1, mode='max'
                                 # , save_best_only=True
                                 )
    # Save model networks
    json_string = model.to_json(indent=4)
    open('./runs/%s/model_pickle.json' % timestamp, 'w').write(json_string)

    # Feed model
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # sgd = SGD(lr=1e-4)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
    # pl_callback = keras.callbacks.ProgbarLogger(count_mode='samples')
    tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHES, verbose=1, callbacks=[checkpoint, tb_callback],
              validation_data=(X_test, y_test))  # starts training
    model.save_weights('./runs/%s/model_1_weights_sub.h5' % timestamp)


def contine_training(timestamp=0, initial_epoch=0):
    # Load model
    json_string = open('./runs/{}/model_pickle.json'.format(str(timestamp)), 'r').read()
    model = models.model_from_json(json_string)

    # Load weights
    weights_file_list = sorted(os.listdir('./runs/{}'.format(str(timestamp))))
    weights_file = weights_file_list[0] if weights_file_list[0] is 'model_1_weights_sub.h5' else weights_file_list[-1]
    model.load_weights('./runs/{}/{}'.format(str(timestamp), weights_file))

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # Continue training
    checkpoint = ModelCheckpoint('./runs/%s/weights.{epoch:03d}-{val_acc:.4f}.hdf5' % timestamp, monitor='val_acc',
                                 verbose=1, mode='max'
                                 # , save_best_only=True
                                 )
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # sgd = SGD(lr=1e-4)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
    tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHES, verbose=1,
              callbacks=[checkpoint, tb_callback],
              validation_data=(X_test, y_test), initial_epoch=initial_epoch)  # starts training
    model.save_weights('./runs/%s/model_1_weights_sub.h5' % timestamp)


if __name__ == '__main__':
    train()
    # contine_training(1494696030, 211)
