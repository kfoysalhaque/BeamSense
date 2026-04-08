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

import numpy as np
from tensorflow import keras
import pandas as pd
from multiprocessing import Pool, cpu_count
import scipy.io as spio
import os

# I keep the default temporal window size used by the data generator.
window_size = 10


def read_mat(dir, file, windowsize):
    # I load one `.mat` sample and map the filename prefix to its class index.
    data = spio.loadmat(os.path.join(dir, file))
    if file[0] == "A":
        angle_data = data["bf_matrix"]
        label = 0
    elif file[0] == "B":
        angle_data = data["bf_matrix"]
        label = 1
    elif file[0] == "C":
        angle_data = data["bf_matrix"]
        label = 2
    elif file[0] == "D":
        angle_data = data["bf_matrix"]
        label = 3
    elif file[0] == "E":
        angle_data = data["bf_matrix"]
        label = 4
    elif file[0] == "F":
        angle_data = data["bf_matrix"]
        label = 5
    if file[0] == "G":
        angle_data = data["bf_matrix"]
        label = 6
    elif file[0] == "H":
        angle_data = data["bf_matrix"]
        label = 7
    elif file[0] == "I":
        angle_data = data["bf_matrix"]
        label = 8
    elif file[0] == "J":
        angle_data = data["bf_matrix"]
        label = 9
    elif file[0] == "K":
        angle_data = data["bf_matrix"]
        label = 10
    elif file[0] == "L":
        angle_data = data["bf_matrix"]
        label = 11
    elif file[0] == "M":
        angle_data = data["bf_matrix"]
        label = 12
    elif file[0] == "N":
        angle_data = data["bf_matrix"]
        label = 13
    elif file[0] == "O":
        angle_data = data["bf_matrix"]
        label = 14
    elif file[0] == "P":
        angle_data = data["bf_matrix"]
        label = 15
    elif file[0] == "Q":
        angle_data = data["bf_matrix"]
        label = 16
    elif file[0] == "R":
        angle_data = data["bf_matrix"]
        label = 17
    elif file[0] == "S":
        angle_data = data["bf_matrix"]
        label = 18
    elif file[0] == "T":
        angle_data = data["bf_matrix"]
        label = 19

    return angle_data, label


class DataGenerator(keras.utils.Sequence):
    """Data generator to load data from batches"""

    def __init__(
        self,
        dataset_path,
        dataset_csv,
        num_classes=20,
        chunk_shape=(window_size, 234, 4),
        batchsize=64,
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
        # I load the file list and labels from the split CSV.
        df = pd.read_csv(dataset_csv)
        self.dataset_path = dataset_path
        self.batchsize = batchsize
        self.datalist = df["filename"]
        self.labels = df["label"]
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.windowsize = chunk_shape[0]
        self.length = chunk_shape[1]
        self.height = chunk_shape[2]
        self.to_categorical = to_categorical

        # I initialize and optionally shuffle index order for batching.
        self.indexes = np.arange(len(self.labels))
        np.random.shuffle(self.indexes)
        self.on_epoch_end()

        return

    def __len__(self):
        """Denote the number of batches"""
        return int(np.floor(len(self.labels) / self.batchsize))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        indexes = self.indexes[idx * self.batchsize:(idx + 1) * self.batchsize]
        X, y = self.__load_batch(indexes)
        return X, y

    def on_epoch_end(self):
        """Update indexes after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __load_batch(self, indexes):
        """Read new batch of data """
        # I assemble one batch by loading each sample from disk.
        batch_data = np.empty((self.batchsize, self.windowsize, self.length, self.height))
        batch_label = np.empty(self.batchsize, dtype=int)
        for i, k in enumerate(indexes):
            batch_data[i], batch_label[i] = read_mat(self.dataset_path, self.datalist[k], self.windowsize)
        if self.to_categorical:
            # I convert integer labels to one-hot encoding when requested.
            batch_label = keras.utils.to_categorical(batch_label, num_classes=20)
        return batch_data, batch_label
