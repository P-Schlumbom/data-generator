# Implementation based on SURF descriptors as described in: Speeded-Up Robust Features (SURF) (Bay et al. 2006)
# Note that this computes UPRIGHT SURF features (i.e. U-SURF), so the overall orientation of the features is not considered
import numpy as np
from scipy import signal # for convolutions

from converters.general_functions import gaussian_2D, normalise

def get_4D_vector(x, y, dx_win, dy_win, step_size):
    x_sub = dx_win[y:y+step_size, x:x+step_size]
    y_sub = dy_win[y:y+step_size, x:x+step_size]
    return np.asarray([np.sum(x_sub), np.sum(np.abs(x_sub)), np.sum(y_sub), np.sum(np.abs(y_sub))])

def compute_SURF_descriptor(x, y, gauss_dx, gauss_dy, size=20):
    step_size = size // 4

    dx_sub = gauss_dx[y:y + size, x:x + size]
    dy_sub = gauss_dy[y:y + size, x:x + size]

    descriptor = np.asarray([])

    for i in range(0, size, step_size):
        for j in range(0, size, step_size):
            descriptor = np.concatenate((descriptor, get_4D_vector(j, i, dx_sub, dy_sub, step_size)))
    return descriptor

def generate_SURF_image(im, s=1):
    """
    Convert mxnx3 input image into oxpx64 array of SURF features, where o = m//20s, p = n//20s. A SURF feature vector is created for each 20sx20s window in the image.
    :param im: numpy array of the input image of shape mxnx3
    :param s: int, scaling factor. Default is 1 ( = 20x20 kernel size)
    :return: numpy array of shape oxpx64
    """

    haar_x = np.ones((2 * s, 2 * s))
    haar_x[:, 0:s] *= -1
    haar_y = haar_x.T
    dx = signal.convolve2d(im, haar_x, 'same')
    dy = signal.convolve2d(im, haar_y, 'same')

    kernel_size = 20 * s
    kernel = gaussian_2D(kernel_size, sigma=3.3 * s)
    kern_grid = np.tile(kernel, (im.shape[0] // kernel_size, im.shape[1] // kernel_size))

    new_shape = (im.shape[0] // kernel_size * kernel_size, im.shape[1] // kernel_size * kernel_size)
    gauss_dx = dx.copy().astype(float)[:new_shape[0], :new_shape[1]]
    gauss_dx *= kern_grid
    gauss_dy = dy.copy().astype(float)[:new_shape[0], :new_shape[1]]
    gauss_dy *= kern_grid

    SURF_image = np.zeros((im.shape[0] // kernel_size, im.shape[1] // kernel_size, 64))

    for i in range(SURF_image.shape[0]):
        for j in range(SURF_image.shape[1]):
            x, y = (j * kernel_size), (i * kernel_size)
            SURF_image[i, j] = normalise(compute_SURF_descriptor(x, y, gauss_dx, gauss_dy, kernel_size))

    return SURF_image

def get_SURF_image(im, s=1):
    """
    Process an image. If the input image has k>1 channels (i.e. not greyscale) then each channel will be processed separately and the resulting arrays concatenated to create a oxpx(k*64) image
    :param im: numpy array of shape nxmxk
    :param s: int, scaling factor. Default is 1 ( = 20x20 kernel size)
    :return: oxpx64k numpy array of SURF feature vectors sampled from the image
    """
    try:
        im_depth = im.shape[2]
    except:
        im_depth = 1

    if im_depth > 1:
        base = generate_SURF_image(im[:,:,0], s)
        for i in range(1, im_depth):
            new = generate_SURF_image(im[:,:,i], s)
            base = np.concatenate((base, new), axis=2)
        return base
    return generate_SURF_image(im, s)