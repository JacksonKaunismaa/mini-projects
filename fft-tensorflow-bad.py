import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import color
from skimage import io
np.set_printoptions(precision=4, suppress=True)

def sin2d(a,h,k,x,y):
    return a * np.sin(h*x + k*y)

def cos2d(a,h,k,x,y):
    return a * np.cos(h*x + k*y)

def mk_img(freqs, pixels=256):
    # list of tuple of (a, h, k) such that the image is the sum of a*sin(h*x + k*y) for all freqs. x and y range from -pi to pi
    coords = np.mgrid[0:pixels,0:pixels].swapaxes(0,2).swapaxes(0,1).astype(np.float32)
    coords /= pixels
    coords *= 2*np.pi
    coords -= np.pi
    img = np.zeros((pixels, pixels))
    for a,h,k in freqs:
        for i,row in enumerate(coords):
            for j,coord in enumerate(row):
                x,y = coords[i,j]
                img[i,j] += cos2d(a,h,k,x,y)
    return img.T


def scale_img(img):
    im_cpy = img.copy()
    im_cpy -= im_cpy.min()
    im_cpy /= im_cpy.max()
    im_cpy *= 255.
    return im_cpy.astype(np.uint8)


def show_img(img):
    im = Image.fromarray(scale_img(img), "L")
    im.show()

def load_img(name):
    im = Image.open(name)
    img = np.array(im).astype(np.float32)
    return img

def load_as_grey(name):
    grey_load = color.rgb2grey(io.imread(name))
    grey_load *= 255.
    return grey_load


def load_and_fft(name):
    load_im = load_img(name)
    try:
        show_img(load_im)
    except ValueError:
        load_im = load_as_grey(name)
        show_img(load_im)
    im_fft = tf.spectral.rfft2d(load_im).numpy()
    show_img(im_fft)

#my_freq = [(1, 10, 0), (1, 0, 1)]
#my_img = mk_img(my_freq, pixels=256)
#show_img(my_img)
#print(my_img)
tf.enable_eager_execution()
#load_and_fft("./lights-bw.jpg")
#load_and_fft("./globe.jpeg")
load_and_fft("./flower.jpeg")

#fft = tf.spectral.rfft2d(my_img).numpy()
#print(fft)
#show_img(fft)
