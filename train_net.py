import os
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l1
from keras.callbacks import EarlyStopping, History
from sklearn.cross_validation import train_test_split
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt

def get_data(file, noise_amt=0.1):
    data_file = h5py.File(file, 'r')
    images = np.mean(np.transpose(data_file['images'][:], (0, 3, 1, 2)), axis=1)
    images = images.reshape(-1, 1, 32, 32)
    noise = np.zeros(images.size)
    noise_amount = noise_amt*noise.size
    indices = np.random.randint(0, images.size, size=int(noise_amount))
    noise[indices] = 1.0
    images = images + noise.reshape(images.shape)
    labels = np_utils.to_categorical(data_file['labels'][:])
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1234)
    return X_train, y_train, X_test, y_test

def get_face_data(file='olivettifaces.mat'):
    import scipy.io as sio
    mat = sio.loadmat(file)
    faces = mat['faces'].T.reshape(-1, 64, 64)
    faces = np.transpose(faces, (0, 2, 1))
    faces = faces.reshape(-1, 1, 64, 64)
    labels = np_utils.to_categorical(np.array(range(len(faces)))/10)

    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.1, random_state=1234, stratify=labels)
    return X_train, y_train, X_test, y_test

def build_model(num_filters=25, nb_classes=10, img_channels=1, img_rows=32, img_cols=32, dropout=0.0):
    model = Sequential()
    model.add(Convolution2D(num_filters, 7, 7, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((12,12)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

def plot_data_sample(data, directory, n=64):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    print(data.shape)
    num_samples = n
    grid_size = int(np.sqrt(n))
    plt.clf()
    plt.figure(figsize=(4.2, 4))
    plt.hold(True)
    for i,j in itertools.product(range(grid_size), range(grid_size)):
        ax = plt.subplot(grid_size, grid_size, grid_size*i + j + 1)
        img = np.transpose(data[i*grid_size + j], (1, 2, 0))
        if img.shape[2] == 1:
            img = img.reshape(img.shape[0], img.shape[1])

        plt.imshow(img, interpolation='none', cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.savefig(os.path.join(directory, 'sample.png'), format='png')

def record_results(history, model, location):
    if not os.path.isdir(location):
        os.makedirs(location)
    history_serial = {}
    history_serial['epoch'] = history.epoch
    history_serial['history'] = history.history
    history_string = json.dumps(history_serial)
    json_string = model.to_json()

    open(os.path.join(location, 'architecture.json'), 'w').write(json_string)
    open(os.path.join(location, 'history.json'), 'w').write(history_string)
    model.save_weights(os.path.join(location, 'weights.hdf5'), overwrite=True)

def make_plots(hist, model, directory, name):

    if not os.path.isdir(directory):
        os.path.makedirs(directory)

    filters = model.get_weights()[0]
    a = plt.figure(figsize=(4.2, 4))
    filters = filters - np.min(filters, axis=0)
    filters = filters / np.max(filters, axis=0)
    plt.hold(True)
    plt.title(name + ' filters')
    num_filters = filters.shape[0]
    grid_size = int(np.sqrt(filters.shape[0]))
    for i,j in itertools.product(range(grid_size), range(grid_size)):
        ax = plt.subplot(grid_size, grid_size, grid_size*i + j + 1)
        plt.xticks(())
        plt.yticks(())
        filter_swapped = np.transpose(filters[i*grid_size + j], (1, 2, 0))
        if filter_swapped.shape[2] == 1:
            filter_swapped = np.mean(filter_swapped, axis=2).reshape(filter_swapped.shape[0], filter_swapped.shape[1])
        plt.imshow(filter_swapped, interpolation='nearest', cmap=plt.cm.gray)
    plt.savefig(os.path.join(directory, 'filters.png'), format='png')
    plt.close(a)

    a = plt.figure(figsize=(16,4))
    plt.hold(True)
    plt.title("Learning Curves")

    x_axis = hist.epoch
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']


    val_loss = hist.history['val_loss']
    train_loss = hist.history['loss']

    plt.subplot(1, 2, 1)
    plt.plot(x_axis, acc, label='Training Accuracy')
    plt.plot(x_axis, val_acc, label='Validation Accuracy')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, train_loss, label='Training Loss')
    plt.plot(x_axis, val_loss, label='Validation Loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(directory, 'learning-curves.png'), format='png')
    plt.close(a)
    
class HistoryUpdate(History):
    def __init__(self, location, name):
        self.location = location
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        History.on_epoch_end(self, epoch, logs)
        record_results(self, self.model, self.location)
        make_plots(self, self.model, self.location, self.name)

def dropout_experiments():

    experiments = [0.2, 0.4, 0.6, 0.8]
    names = ['dropout-0.2', 'dropout-0.4', 'dropout-0.6', 'dropout-0.8']
    X_train, y_train, X_test, y_test = get_data('./data/letters.hdf5', noise_amt=0.1)
    for name, experiment_params in reversed(zip(names, experiments)):
        dir = os.path.join(os.getcwd(), 'results', 'dropout_alpha_25_noisy', name)
        history = HistoryUpdate(dir, name)
        plot_data_sample(X_train, dir, 64)
        print("Running experiment {name}".format(name=name))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model = build_model(dropout=experiment_params)
        hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=200,
                         show_accuracy=True, validation_data=(X_test, y_test),
                         shuffle=True, callbacks=[early_stopping, history])

        #record_results(hist, model, dir)
        #make_plots(hist, model, dir, name)

def face_experiments():
    experiments = [0.2, 0.4, 0.6, 0.8]
    names = ['dropout-0.2', 'dropout-0.4', 'dropout-0.6', 'dropout-0.8']
    X_train, y_train, X_test, y_test = get_face_data()
    for name, experiment_params in zip(names, experiments):
        dir = os.path.join(os.getcwd(), 'results', 'dropout_faces', name)
        plot_data_sample(X_train, dir, 64)
        print("Running experiment {name}".format(name=name))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model = build_model(num_filters=64, dropout=experiment_params, img_rows=64, img_cols=64, nb_classes=40)
        hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=200,
                         show_accuracy=True, validation_data=(X_test, y_test),
                         shuffle=True, callbacks=[early_stopping])

        record_results(hist, model, dir)
        make_plots(hist, model, dir, name)

if __name__ == "__main__":
    dropout_experiments()