import tensorflow as tf
import numpy as np
import collections
import pickle
import os
from PIL import Image

"""After having run the raw_loader.py (found in ../raw/raw_loader.py) you should now have some filled directories of stored images + label files.
This script turns those images into TFRecords for even better storage + use for data loading"""


idx_to_name = {0: "нуль",
               1: "один",
               2: "два",
               3: "три",
               4: "четерые",
               5: "пять",
               6: "шесть",
               7: "семь",
               8: "восемь",
               9: "девять"}

normie_nums = {0: "zero",
               1: "one",
               2: "two",
               3: "three",
               4: "four",
               5: "five",
               6: "six",
               7: "seven",
               8: "eight",
               9: "nine"}


def unpickle(name):
	with open(name, "rb") as p:
		data = pickle.load(p)
	return data

def _floats_feature(val):
    if isinstance(val, collections.Iterable):
        return tf.train.Feature(float_list=tf.train.FloatList(value=val))
    return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))

def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def _int64_feature(val):
    if isinstance(val, collections.Iterable):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=val))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def serialize_example_pyf(tf_img, output_idx, num_name):
	feature = {"img": _floats_feature(tf_img.astype(np.float32).reshape(784)),
            "lbl": _int64_feature(output_idx),
            "name": _bytes_feature(num_name)}
	proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return proto.SerializeToString()

def tf_serialize_example(img, out_idx, num_name):
    tf_string = tf.py_func(serialize_example_pyf, (img, out_idx, num_name), tf.string)
    return tf.reshape(tf_string, ())

def load_labels(name):
    return unpickle(name)

def load_everything(dir_name, lbls):
    img_files = os.listdir(dir_name)
    img_files.sort(key=int)
    total_img_files = [f"./{dir_name}/{imname}" for imname in img_files]
    return list(map(unpickle, total_img_files)), lbls, [idx_to_name[x] for x in lbls]

def show_img(x):
    img = Image.fromarray(x.astype(np.uint8), "L")
    img.show()


train_labels = load_labels("./train.labels")
test_labels = load_labels("./test.labels")
train_data = load_everything("./train_images", train_labels)
test_data = load_everything("./test_images", test_labels)
assert (isinstance(train_labels,list) and isinstance(train_data,tuple)          \
        and isinstance(train_data[0],list) and isinstance(train_data[1],list)  \
        and isinstance(train_data[2],list)), "some type errors"

tf.enable_eager_execution()
rand3 = np.random.randint(0, 2000, 3)
print("Hopefully these are correct??")
show_img(train_data[0][rand3[0]])
show_img(train_data[0][rand3[1]])
show_img(train_data[0][rand3[2]])
print(train_data[1][rand3[0]])
print(train_data[1][rand3[1]])
print(train_data[1][rand3[2]])
print("Creating base Datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
print("Mapping Datasets...")
train_dataset = train_dataset.map(tf_serialize_example)
test_dataset = test_dataset.map(tf_serialize_example)
print("Creating writers...")
train_writer = tf.data.experimental.TFRecordWriter("train.tfr")
test_writer = tf.data.experimental.TFRecordWriter("test.tfr")
print("Writing datasets to tfr files...")
train_writer.write(train_dataset)
test_writer.write(test_dataset)




