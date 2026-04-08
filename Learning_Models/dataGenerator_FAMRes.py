"""
    Copyright (C) 2023 Khandaker Foysal Haque
    contact: haque.k@northeastern.edu
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os

import numpy as np
import pandas as pd
import scipy.io as spio
from tensorflow import keras

window_size = 10


def read_mat(dir1, dir2, dir3, file, windowsize):
    # I load the same filename from all three station directories.
    data1 = spio.loadmat(os.path.join(dir1, file))
    data2 = spio.loadmat(os.path.join(dir2, file))
    data3 = spio.loadmat(os.path.join(dir3, file))

    angle_data1 = data1["bf_matrix"]
    angle_data2 = data2["bf_matrix"]
    angle_data3 = data3["bf_matrix"]

    if file[0] == "A":
        label = 0
    elif file[0] == "B":
        label = 1
    elif file[0] == "C":
        label = 2
    elif file[0] == "D":
        label = 3
    elif file[0] == "E":
        label = 4
    elif file[0] == "F":
        label = 5
    if file[0] == "G":
        label = 6
    elif file[0] == "H":
        label = 7
    elif file[0] == "I":
        label = 8
    elif file[0] == "J":
        label = 9
    elif file[0] == "K":
        label = 10
    elif file[0] == "L":
        label = 11
    elif file[0] == "M":
        label = 12
    elif file[0] == "N":
        label = 13
    elif file[0] == "O":
        label = 14
    elif file[0] == "P":
        label = 15
    elif file[0] == "Q":
        label = 16
    elif file[0] == "R":
        label = 17
    elif file[0] == "S":
        label = 18
    elif file[0] == "T":
        label = 19

    angle_data = np.concatenate((angle_data1, angle_data2, angle_data3), axis=2) / 180
    return angle_data, label


class DataGenerator(keras.utils.Sequence):
    """Data generator to load data from h5"""

    def __init__(
        self,
        data_path1,
        data_path2,
        data_path3,
        dataset_csv,
        num_classes=20,
        chunk_shape=(window_size, 234, 4),
        batchsize=32,
        shuffle=True,
        to_categorical=True,
    ):
        """Initialization
        param:
            dataset_path: the directory to stored .mat files
            dataset_csv: the csv file to store the list of files and labels
            num_classes: number of classes
            chunk_shape: shape of data (number of samples x 2)
        """
        df = pd.read_csv(dataset_csv)
        self.dataset_path1 = data_path1
        self.dataset_path2 = data_path2
        self.dataset_path3 = data_path3
        self.batchsize = batchsize
        self.datalist = df["filename"]
        self.labels = df["label"]
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.windowsize = chunk_shape[0]
        self.length = chunk_shape[1]
        self.height = chunk_shape[2] * 3
        self.to_categorical = to_categorical
        self.indexes = np.arange(len(self.labels))
        np.random.shuffle(self.indexes)
        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches"""
        return int(np.floor(len(self.labels) / self.batchsize))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        indexes = self.indexes[idx * self.batchsize : (idx + 1) * self.batchsize]
        X, y = self.__load_batch(indexes)
        return X, y

    def on_epoch_end(self):
        """Update indexes after each epoch"""
        if self.shuffle == True:
            # I reshuffle indexes between epochs when shuffle is enabled.
            np.random.shuffle(self.indexes)

    def __load_batch(self, indexes):
        """Read new batch of data"""
        batch_data = np.empty((self.batchsize, self.windowsize, self.length, self.height))
        batch_label = np.empty(self.batchsize, dtype=int)
        for i, k in enumerate(indexes):
            batch_data[i], batch_label[i] = read_mat(
                self.dataset_path1,
                self.dataset_path2,
                self.dataset_path3,
                self.datalist[k],
                self.windowsize,
            )
        if self.to_categorical:
            batch_label = keras.utils.to_categorical(batch_label, num_classes=20)
        return batch_data, batch_label
