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

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as spio
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataGenerator_FAMRes import DataGenerator

window = 10

learning_rate = 0.01
meta_step_size = 0.25  # 1
inner_batch_size = 20
meta_iters = 500
inner_iters = 4
eval_interval = 1
train_shots = 1
classes = 20


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("pretrained_env", help="Pretrained Scenario")
    parser.add_argument("testing_env", help="Testing Scenario")
    parser.add_argument("sample_time", help="sample_time")
    parser.add_argument("model_save", help="Name of the model")

    args = parser.parse_args()

    pretrained_env = args.pretrained_env
    testing_env = args.testing_env
    sample_time = args.sample_time
    model_save = args.model_save
    window_size = window

    class Dataset:
        def __init__(
            self,
            dir1,
            dir2,
            dir3,
            csv,
            windowsize=10,
            classes=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"],
        ):
            # I keep one dataframe loaded and reuse it for all meta-learning batches.
            self.df = pd.read_csv(csv)
            self.dir1 = dir1
            self.dir2 = dir2
            self.dir3 = dir3
            self.classes = classes
            self.windowsize = windowsize

        def get_mini_dataset(self, batch_size, repetitions, shots, num_classes, split=False, to_categorical=True):
            # I sample a class-balanced mini set for the current inner-loop update.
            dfs = []
            np.random.seed(111)
            shot_classes = np.random.choice(self.classes, size=num_classes, replace=False)
            for shot_class in shot_classes:
                dfs.append(self.df.loc[self.df["label"] == shot_class].sample(n=shots))
            df_mini = pd.concat(dfs)

            mini_dataset_data = np.empty((num_classes * shots, self.windowsize, 234, 12))
            mini_dataset_label = np.empty(num_classes * shots)
            for i, e in enumerate(df_mini["filename"]):
                mini_dataset_data[i], mini_dataset_label[i] = self.read_mat(self.dir1, self.dir2, self.dir3, e)

            if to_categorical:
                mini_dataset_label = keras.utils.to_categorical(mini_dataset_label, num_classes=len(self.classes))

            mini_dataset = tf.data.Dataset.from_tensor_slices(
                (mini_dataset_data.astype(np.float32), mini_dataset_label.astype(np.int32))
            )
            mini_dataset = mini_dataset.shuffle(df_mini.size).batch(batch_size).repeat(repetitions)
            return mini_dataset

        def read_mat(self, dir1, dir2, dir3, file):
            data1 = spio.loadmat(os.path.join(dir1, file))
            data2 = spio.loadmat(os.path.join(dir2, file))
            data3 = spio.loadmat(os.path.join(dir3, file))

            angle_data1 = data1["bf_matrix"] / 180
            angle_data2 = data2["bf_matrix"] / 180
            angle_data3 = data3["bf_matrix"] / 180

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

            angle_data = np.concatenate((angle_data1, angle_data2, angle_data3), axis=2)
            return angle_data, label

    # I build all dataset paths in one place so I can switch environments quickly.
    data_path = "../Data/"
    data_proc = "processed_dataset"

    data_env1_dir1 = os.path.join(data_path, testing_env, data_proc, "9C", "beamf_angles")
    data_env1_dir2 = os.path.join(data_path, testing_env, data_proc, "25", "beamf_angles")
    data_env1_dir3 = os.path.join(data_path, testing_env, data_proc, "89", "beamf_angles")

    micro_csv_name = "mini_set_" + sample_time + "s.csv"
    meta_csv = os.path.join(data_env1_dir1, micro_csv_name)
    meta_dataset = Dataset(data_env1_dir1, data_env1_dir2, data_env1_dir3, meta_csv, windowsize=10)

    pretrained_model = (
        "../Data/" + pretrained_env + "/processed_dataset/beamf_angles/" + pretrained_env + "_combined_0.1.h5"
    )
    model = keras.models.load_model(pretrained_model)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # I first run the Reptile-style adaptation loop.
    for meta_iter in range(meta_iters):
        frac_done = meta_iter / meta_iters
        cur_meta_step_size = (1 - frac_done) * meta_step_size
        old_vars = model.get_weights()
        mini_dataset = meta_dataset.get_mini_dataset(inner_batch_size, inner_iters, train_shots, classes)

        for images, labels in mini_dataset:
            with tf.GradientTape() as tape:
                preds = model(images)
                loss = keras.losses.categorical_crossentropy(labels, preds)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        new_vars = model.get_weights()
        for var in range(len(new_vars)):
            new_vars[var] = old_vars[var] + ((new_vars[var] - old_vars[var]) * cur_meta_step_size)
        model.set_weights(new_vars)

    window_size = 10

    data_env2_dir1 = os.path.join(data_path, testing_env, data_proc, "9C", "beamf_angles")
    data_env2_dir2 = os.path.join(data_path, testing_env, data_proc, "25", "beamf_angles")
    data_env2_dir3 = os.path.join(data_path, testing_env, data_proc, "89", "beamf_angles")
    model_dir = os.path.join(data_path, testing_env, data_proc, "beamf_angles", model_save)

    val_csv_name = "val_set_" + sample_time + "s.csv"
    large_csv_name = "large_set.csv"

    tr_csv_env2 = os.path.join(data_env2_dir1, micro_csv_name)
    val_csv_env2 = os.path.join(data_env2_dir1, val_csv_name)
    test_csv_env2 = os.path.join(data_env2_dir1, large_csv_name)

    train_gen = DataGenerator(data_env2_dir1, data_env2_dir2, data_env2_dir3, tr_csv_env2, batchsize=32)
    val_gen = DataGenerator(data_env2_dir1, data_env2_dir2, data_env2_dir3, val_csv_env2, batchsize=32)
    test_gen = DataGenerator(data_env2_dir1, data_env2_dir2, data_env2_dir3, test_csv_env2, batchsize=32, shuffle=False)

    learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    checkpoint = ModelCheckpoint(model_dir, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor="val_loss", min_delta=0.05, patience=40, verbose=1)

    model.compile(optimizer=keras.optimizers.Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.set_weights(new_vars)
    history = model.fit(
        x=train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=[learning_rate_reduction, checkpoint],
        verbose=1,
    )

    plt.plot(history.history["accuracy"], label="Training acc")
    plt.plot(history.history["val_accuracy"], label="Validation acc")
    plt.legend()
    plt.savefig((os.path.join(data_path, testing_env, data_proc, "beamf_angles", "train_val_accuracy.png")), dpi=300)
    plt.show()

    print("The validation accuracy is :", history.history["val_accuracy"])
    print("The training accuracy is :", history.history["accuracy"])
    print("The validation loss is :", history.history["val_loss"])
    print("The training loss is :", history.history["loss"])

    final_loss, final_accuracy = model.evaluate(test_gen)
    print("Final Loss: {}, Final Accuracy: {}".format(final_loss, final_accuracy))

    labels = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"}
    import seaborn as sns

    Y_pred = model.predict(test_gen)
    Y_pred = np.argmax(Y_pred, axis=1)

    Y = test_gen.labels[test_gen.indexes]
    Y_true = np.zeros(len(Y))

    # I convert alphabetic labels to numeric ids so sklearn metrics can use them directly.
    for i, e in enumerate(Y):
        if e == "A":
            Y_true[i] = 0
        elif e == "B":
            Y_true[i] = 1
        elif e == "C":
            Y_true[i] = 2
        elif e == "D":
            Y_true[i] = 3
        elif e == "E":
            Y_true[i] = 4
        elif e == "F":
            Y_true[i] = 5
        elif e == "G":
            Y_true[i] = 6
        elif e == "H":
            Y_true[i] = 7
        elif e == "I":
            Y_true[i] = 8
        elif e == "J":
            Y_true[i] = 9
        elif e == "K":
            Y_true[i] = 10
        elif e == "L":
            Y_true[i] = 11
        elif e == "M":
            Y_true[i] = 12
        elif e == "N":
            Y_true[i] = 13
        elif e == "O":
            Y_true[i] = 14
        elif e == "P":
            Y_true[i] = 15
        elif e == "Q":
            Y_true[i] = 16
        elif e == "R":
            Y_true[i] = 17
        elif e == "S":
            Y_true[i] = 18
        else:
            Y_true[i] = 19
    print(Y_true)

    cm = confusion_matrix(Y_true[: len(Y_pred)], Y_pred, normalize="true")
    plt.figure(figsize=(24, 24))
    ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=labels, yticklabels=labels)
    ax.set_ylabel("Actual", fontsize=20)
    ax.set_xlabel("Predicted", fontsize=20)
    plt.savefig((os.path.join(data_path, testing_env, data_proc, "beamf_angles", "conf_matrix_FAMRes.png")), dpi=300)

    print("\nAccuracy: {:.4f}\n".format(accuracy_score(Y_true[: len(Y_pred)], Y_pred)))
