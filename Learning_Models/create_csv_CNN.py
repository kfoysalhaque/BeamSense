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
import os
import csv

# I set the dataset location for the station/scenario I want to process.
station = "9C/beamf_angles"
Test = "Classroom_All"
proc_dir = "processed_dataset"

# I fix the random seed so the train/val/test split stays reproducible.
np.random.seed(111)
data_pa = "../Data"

data_path = os.path.join(data_pa, Test, proc_dir, station)

train_csv = os.path.join(data_path, "train_set.csv")
val_csv = os.path.join(data_path, "val_set.csv")
test_csv = os.path.join(data_path, "test_set.csv")


def custom_sort_key(filename):
    # I sort by sample index and offset selected person IDs to keep ordering stable.
    _, person, _, index = filename.split("_")
    value = int(index[:-4])
    if person == "72":
        value += 500000
    elif person == "73":
        value += 1000000
    return value

# I open output CSV files and write headers for each split.
train_csv = open(train_csv, "w", newline="")
val_csv = open(val_csv, "w", newline="")
test_csv = open(test_csv, "w", newline="")
fieldnames = ["filename", "label"]
writer_train = csv.DictWriter(train_csv, fieldnames=fieldnames)
writer_train.writeheader()
writer_val = csv.DictWriter(val_csv, fieldnames=fieldnames)
writer_val.writeheader()
writer_test = csv.DictWriter(test_csv, fieldnames=fieldnames)
writer_test.writeheader()

# I scan each batch folder, sort files, and split records into train/val/test.
for root, dirs, files in os.walk(data_path):
    if root[-5:] == "batch":
        for file in sorted(files, key=lambda x: custom_sort_key(x)):
            filename = os.path.join(root[-7:], file)
            label = root[-7]
            rand = np.random.rand(1)
            if rand < 0.7:
                writer_train.writerow({"filename": filename, "label": label})
            elif rand < 0.85:
                writer_val.writerow({"filename": filename, "label": label})
            else:
                writer_test.writerow({"filename": filename, "label": label})
