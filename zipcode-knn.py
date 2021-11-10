import knn
from PIL import Image
import numpy as np

def load_data(fname, im_size=16):
    with open(fname, "r") as f:
        raw = [l.split() for l in f.readlines()]
    classifications = [int(float(x[0])) for x in raw]
    str_imgs = [x[1:] for x in raw]
    f_imgs = [np.array([float(x) for x in str_img]).reshape(im_size*im_size) for str_img in str_imgs]
    return np.array(classifications), np.array(f_imgs)


def show_image(an_img):
    an_img += 1
    an_img /= 2.0
    an_img *= 5
    im = Image.fromarray(np.uint8(an_img.reshape(16,16)), "L")
    im.show()

def test_selection(an_model, selection):
    sel_idxs = np.where(test_labels == selection)
    test_labels_final = test_labels[sel_idxs]
    test_imgs_final = test_imgs[sel_idxs]
    correct = 0
    total = 0
    for x_img, actual_label in zip(test_imgs_final, test_labels_final):
        if an_model.classify(x_img) == actual_label:
            correct += 1
        total += 1
    print(f"Test accuracy on all the {selection}'s was {correct} / {total} =", float(correct)/total)


def train_selection(an_model, selection):
    sel_idxs = np.where(train_labels == selection)
    train_labels_final = train_labels[sel_idxs]
    train_imgs_final = train_imgs[sel_idxs]
    correct = 0
    total = 0
    for x_img, actual_label in zip(train_imgs_final, train_labels_final):
        if an_model.classify(x_img) == actual_label:
            correct += 1
        total += 1
    print(f"Train accuracy on all the {selection}'s was {correct} / {total} =", float(correct)/total)


train_labels, train_imgs = load_data("./datasets/zip.train")
test_labels, test_imgs = load_data("./datasets/zip.test")
k_val = int(input("k value: "))
model = knn.KNNClassifier(k_val)
model.input_training(train_imgs, train_labels)
while True:
    sel_num = int(input("Num to test: "))
    train_selection(model, sel_num)
    test_selection(model, sel_num)
