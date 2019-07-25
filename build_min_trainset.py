import os, os.path
import shutil
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

train_dir = os.path.join (dir_path, "open-image-dataset/train")
train_min_dir = os.path.join (dir_path, "open-image-dataset/train_min")

train_files = os.listdir(train_dir)
number_of_train_file = len(train_files)

copy_index = random.sample(range(number_of_train_file), number_of_train_file // 5)

for i in copy_index:
    shutil.copy(os.path.join(train_dir, train_files[i]), train_min_dir)


print("Done creating train_min dataset")
print("Number of train files: ", len(os.listdir(train_dir)))
print("Number of train_min files: ", len(os.listdir(train_min_dir)))

