import numpy as np
from os import listdir
from keras.models import load_model
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
from architectures.capsule_net.simple_capsnet import create_simple_capsnet_model

def get_3channel_im(path):
    im = imread(path)
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    if im.shape[2] == 1:  # i.e. if it's a greyscale image, convert it into a 3-channel image
        im = np.concatenate((im, im, im), axis=2)
    return im

#datapath = "testing/imsets/displaced-atomic-eg/"
#outpath = "testing/imsets/"
#datapath = "testing/activation gid/reconstructions/"
#outpath = "testing/activation gid/"
datapath = "testing/activation gid/raw encodings/"
outpath = "testing/activation gid/"
imnames = listdir(datapath)
print(imnames)

#im = get_3channel_im(datapath + imnames[0])

#im_shape = im.shape

"""#grid
size = int(np.sqrt(len(imnames))) # number of ims on a side

base = np.zeros((im_shape[0] * size, im_shape[1] * size, im_shape[2]))
print(base.shape)

for i in range(len(imnames)):
    x, y = (i%size) * im_shape[1], (i//size) * im_shape[0]

    base[y:y + im_shape[0], x:x + im_shape[1]] = get_3channel_im(datapath + imnames[i])

base = np.ndarray.astype(base, int)"""

#row
"""base = np.zeros((im_shape[0], im_shape[1] * len(imnames)))
for i in range(len(imnames)):
    base[:,im_shape[1] * i:(im_shape[1] * i) + im_shape[1]] = imread(datapath + imnames[i])"""

#npy matrices
base = np.load(datapath + imnames[0])
for i in range(1,len(imnames)):
    if i%3 == 0:
        base = np.concatenate((base, np.zeros((80,5))), axis=1)
    base = np.concatenate((base,np.load(datapath + imnames[i])),axis=1)

plt.imshow(base)
plt.show()

imsave(datapath[:-1] + ".png", base)