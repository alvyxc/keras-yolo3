import os, os.path
import shutil
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

train_dir = os.path.join (dir_path, "open-image-dataset/train")
test_dir = os.path.join (dir_path, "open-image-dataset/test")

train_files = os.listdir(train_dir)
number_of_train_file = len(train_files)

copy_index = random.sample(range(number_of_train_file), number_of_train_file // 10)

for i in copy_index:
    shutil.move(os.path.join(train_dir, train_files[i]), test_dir)


print("Done creating test dataset")
print("Number of train files: ", len(os.listdir(train_dir)))
print("Number of test files: ", len(os.listdir(test_dir)))

