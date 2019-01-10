# Implementation based on Lowe's SIFT features as described in: Distinctive Image Features from Scale-Invariant Keypoints (Lowe 2004)
# Note that this computes UPRIGHT SIFT features (i.e. U-SIFT), so the overall orientation of the features is not considered
import numpy as np
from scipy import signal # for convolutions

from converters.general_functions import gaussian_2D, normalise


#-----------------------------NOMRMALISATION---------------------------------#
def normalise_feature(vector):
    vector =normalise(vector)
    vector[vector > 0.2] = 0.2
    vector = normalise(vector)
    return vector

#------------------------------HISTOGRAM-------------------------------------#
def get_histogram(x, y, angles, magnitudes, size=4):
    # window = angles[y:y+size, x:x+size]

    HOG = np.zeros((8,))

    for i in range(y, y + size):
        for j in range(x, x + size):
            HOG[angles[i, j]] += magnitudes[i, j]

    return HOG

#------------------------------GET SIFT IMAGE-------------------------------#
def compute_SIFT_descriptor(x, y, angles, magnitudes, size=16):
    step_size = size // 4

    mag_sub = magnitudes[y:y + size, x:x + size]
    angle_sub = angles[y:y + size, x:x + size]

    descriptor = np.asarray([])

    for i in range(0, size, step_size):
        for j in range(0, size, step_size):
            descriptor = np.concatenate((descriptor, get_histogram(j, i, angle_sub, mag_sub, step_size)))
    return descriptor

def generate_SIFT_image(im, kernel_size=32):
    """
    Convert mxnx1 input image into oxpx128 array of SIFT features, where o = m//kernel_size, p = n//kernel_size. A SIFT feature vector is created for each kernel_sizexkernel_size window in the image.
    :param im: numpy array representation of the input image in the shape mxnx1
    :param kernel_size: int, size (along the side) of the region of interest for which a SIFT feature is generated
    :return: oxpx128 numpy array of SIFT feature vectors sampled from the image
    """
    kernel = gaussian_2D(kernel_size, sigma=kernel_size // 2)
    SIFT_image = np.zeros((im.shape[0] // kernel_size, im.shape[1] // kernel_size, 128))
    kern_grid = np.tile(kernel, (SIFT_image.shape[0], SIFT_image.shape[1]))

    new_shape = (im.shape[0] // kernel_size * kernel_size, im.shape[1] // kernel_size * kernel_size)
    new_im = im.copy()
    new_im = new_im.astype(float)[:new_shape[0], :new_shape[1]]

    x_diff = np.asarray([[-1, 0, 1]])
    y_diff = np.asarray([[-1, 0, 1]]).T
    dx = signal.convolve2d(new_im, x_diff, 'same')
    dy = signal.convolve2d(new_im, y_diff, 'same')
    angles = np.arctan2(dy, dx) * 180 / np.pi
    angles = (((angles + 180) % 360) // 45).astype(int)
    magnitudes = np.sqrt(dx ** 2 + dy ** 2) * kern_grid

    for i in range(SIFT_image.shape[0]):
        for j in range(SIFT_image.shape[1]):
            x, y = (j * kernel_size), (i * kernel_size)
            SIFT_image[i, j] = normalise_feature(compute_SIFT_descriptor(x, y, angles, magnitudes, kernel_size))
            # SIFT_image[i, j] = compute_SIFT_descriptor(x, y, kernel_size)
            # magnitudes[y-(kernel_size//2):y+(kernel_size//2), x-(kernel_size//2):x+(kernel_size//2)] *= kernel # the magnitudes are weighted  by the gaussian
    return SIFT_image

def get_SIFT_image(im, kernel_size=32):
    """
    Process an image. If the input image has k>1 channels (i.e. not greyscale) then each channel will be processed separately and the resulting arrays concatenated to create a oxpx(k*128) image
    :param im: numpy array of shape nxmxk
    :param kernel_size: int, size (along the side) of the region of interest for which a SIFT feature is generated
    :return: oxpx128k numpy array of SIFT feature vectors sampled from the image
    """
    try:
        im_depth = im.shape[2]
    except:
        im_depth = 1

    if im_depth > 1:
        base = generate_SIFT_image(im[:,:,0], kernel_size)
        for i in range(1, im_depth):
            new = generate_SIFT_image(im[:,:,i], kernel_size)
            base = np.concatenate((base, new), axis=2)
        return base
    return generate_SIFT_image(im, kernel_size)