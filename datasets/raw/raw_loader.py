import pathlib
from PIL import Image
import pickle
import numpy as np
import math

"""Little script to turn the wierd image format for MNIST from http://yann.lecun.com/exdb/mnist/ to numpy arrays
Unzip and save the four files given there as train_images, train_labels, test_labels, and test_images
You should also save this + the 4 files in an upper folder (maybe called 'raw') and then create a lower folder called 'pkl' where the numpy arrays will be stored, then 2 folders inside pkl
called train_images and test_images (so you now should have made 4 directories + subdirectories)
It saves each image individually as a pickled numpy array (dtype=np.uint8) with a number corresponding to its index label in the appropriate .labels file,
in separate files, but all the labels are in one file (one for test and one for train though)"""

def bytes_to_ints(byte_array):
    return [x for x in byte_array]


def get_an_img(byte_array, offset):
    img_array = np.zeros((28,28)).astype(np.uint8)
    for idx in range(28):
        img_array[idx, :] = bytes_to_ints(byte_array[offset+idx*28:offset+idx*28+28])
    return img_array

def get_all_imgs(byte_array, num, offset):
    # 60000 for train, 10000 for test = num
    img_list = [get_an_img(byte_array, 16+784*n) for n in range(num)]
    return img_list

def display_some_imgs(img_array, idxs):
    img_array = np.array(img_array)
    for im_show in img_array[idxs]:
        im = Image.fromarray(im_show, "L")
        im.show()

def get_labels(byte_array, offset):
    return [x for x in byte_array[offset:]]

with open("train_images", "rb") as p:
    train_b_array = p.read()
with open("test_images", "rb") as p:
    test_b_array = p.read()
with open("train_labels", "rb") as p:
    train_l_array = p.read()
with open("test_labels", "rb") as p:
    test_l_array = p.read()

train = get_all_imgs(train_b_array, 60000, 16)
print("Total num of train images:", len(train))
test = get_all_imgs(test_b_array, 10000, 16)
print("Total num of test images:", len(test))
train_labels = get_labels(train_l_array, 8)
print("Total num of train labels:", len(train_labels))
test_labels = get_labels(test_l_array, 8)
print("Total num of test labels:", len(test_labels))


print("Displaying random examples of train set...")
idx_choices = np.random.choice(10000, size=3)
display_some_imgs(train, idx_choices)
print(np.array(train_labels)[idx_choices])
check1 = input("The above are supposed to be the labels for the displayed images, is this correct (y/n)? ")
if check1.lower() not in ['y', 'yes', 'ya', 'yep']:
    print("Uh oh images didn't load correctly, aborting...")
    quit()

print("Displaying random examples of test set...")
display_some_imgs(test, idx_choices)
print(np.array(test_labels)[idx_choices])
check2 = input("The above are supposed to be the labels for the displayed images, is this correct (y/n)? ")
if check2.lower() not in ['y', 'yes', 'ya', 'yep']:
    print("Uh oh images didn't load correctly, aborting...")
    quit()

def pad(idx, size):
    return "0" * (size - len(str(idx))) + str(idx)

def save_imgs_as_folder(name, data):
    data_size = math.ceil(np.log10(len(data)))
    save_loc = pathlib.Path(f"../pkl/{name}")
    for idx, datum in enumerate(data):
        with open(save_loc / str(idx), "wb") as p:
            pickle.dump(datum, p)

print("Sketchy checks completed, saving data...")
#with open("../pkl/train.images", "wb") as p:
#    pickle.dump(train, p)
#with open("../pkl/test.images", "wb") as p:
#    pickle.dump(test, p)
save_imgs_as_folder("train_images", train)
save_imgs_as_folder("test_images", test)
with open("../pkl/train.labels", "wb") as p:
    pickle.dump(train_labels, p)
with open("../pkl/test.labels", "wb") as p:
    pickle.dump(test_labels, p)
print("All data saved!")
