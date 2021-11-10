import numpy as np
from PIL import Image



IM_SIZE = 512



def show_img(np_arr):
    im_arr = np_arr * 255
    img = Image.fromarray(im_arr.astype(np.uint8), "L")
    img = img.resize((IM_SIZE, IM_SIZE))
    img.show()

conv_filter = np.random.random((1, 1, 64, 64)
conv_filter = np.reshape(conv_filter, [1*1*64, 128])
show_img(conv_filter)
