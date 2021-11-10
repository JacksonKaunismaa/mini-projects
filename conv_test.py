import numpy as np
import tensorflow as tf



conv_str ="""
['0000', '0100', '    ', '1000', '1100', '    ', '    ', '    ', '    ', '0010', '0110', '    ', '1010', '1110', '    ', '    ', '    ', '    ']|
['    ', '0000', '0100', '    ', '1000', '1100', '    ', '    ', '    ', '    ', '0010', '0110', '    ', '1010', '1110', '    ', '    ', '    ']|
['    ', '    ', '0000', '0100', '    ', '1000', '1100', '    ', '    ', '    ', '    ', '0010', '0110', '    ', '1010', '1110', '    ', '    ']|
['    ', '    ', '    ', '0000', '0100', '    ', '1000', '1100', '    ', '    ', '    ', '    ', '0010', '0110', '    ', '1010', '1110', '    ']|
['0001', '0101', '    ', '1001', '1101', '    ', '    ', '    ', '    ', '0011', '0111', '    ', '1011', '1111', '    ', '    ', '    ', '    ']|
['    ', '0001', '0101', '    ', '1001', '1101', '    ', '    ', '    ', '    ', '0011', '0111', '    ', '1011', '1111', '    ', '    ', '    ']|
['    ', '    ', '0001', '0101', '    ', '1001', '1101', '    ', '    ', '    ', '    ', '0011', '0111', '    ', '1011', '1111', '    ', '    ']|
['    ', '    ', '    ', '0001', '0101', '    ', '1001', '1101', '    ', '    ', '    ', '    ', '0011', '0111', '    ', '1011', '1111', '    ']|
['0002', '0102', '    ', '1002', '1102', '    ', '    ', '    ', '    ', '0012', '0112', '    ', '1012', '1112', '    ', '    ', '    ', '    ']|
['    ', '0002', '0102', '    ', '1002', '1102', '    ', '    ', '    ', '    ', '0012', '0112', '    ', '1012', '1112', '    ', '    ', '    ']|
['    ', '    ', '0002', '0102', '    ', '1002', '1102', '    ', '    ', '    ', '    ', '0012', '0112', '    ', '1012', '1112', '    ', '    ']|
['    ', '    ', '    ', '0002', '0102', '    ', '1002', '1102', '    ', '    ', '    ', '    ', '0012', '0112', '    ', '1012', '1112', '    ']"""


init_imgs = np.random.random((1,3,3,2))
init_filt = np.random.random((2,2,2,3))

def custom_vectorize(an_arr, size):
    new_img = np.zeros_like(an_arr)
    for x in range(3):
        for y in range(3):
            for z in range(2):
                new_img[x,y,z] = an_arr[x,y,z]
    return new_img.reshape(size)


def mat_conv(img_in, weight_in):
    r = np.squeeze(init_imgs)
    img_r = np.stack((np.stack(np.split(r,6)[:3]), np.stack(np.split(r,6)[3:])), axis=2)
    conv_mat = np.zeros((12,18))
    for x, row in enumerate(conv_str.split("|")):
        row_values = row[2:-1].replace("'","").split(",")
        for y, value in enumerate(row_values):
            try:
                conv_mat[x,y] = w[int(value[0]), int(value[1]), int(value[2]), int(value[3])]
            except ValueError:
                pass
    our_res = np.matmul(conv_mat, img_r)
    return our_res.reshape((1,2,2,3))




tf.enable_eager_execution()

img = tf.Variable(initial_value=init_imgs)
w = tf.Variable(initial_value=init_filt)
res = tf.nn.conv2d(img, w, padding="VALID", strides=[1,1,1,1]).numpy()
other_res = mat_conv(init_imgs, init_filt)
print(res - other_res)


