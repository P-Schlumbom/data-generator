import numpy as np
from os import listdir, walk, path, makedirs
from sys import argv, stdout
from scipy.misc import imread

from converters.SIFT_generator import get_SIFT_image
from converters.SURF_generator import get_SURF_image
from converters.encoded_generator import Encoder
from converters.caps_generator import CapsuleEncoder

def autoencoder_encode(encoder, im):
    return encoder.get_encoded_image(im)

def capsencoder_encode(encoder, im):
    return encoder.get_caps_encoded_image(im)

refs = {
    'sift' : get_SIFT_image,
    'surf' : get_SURF_image,
    'autoencoder' : autoencoder_encode,
    'capsencoder' : capsencoder_encode
}

inits = {
    'autoencoder' : Encoder,
    'capsencoder' : CapsuleEncoder
}

filepath = argv[1]
try:
    outname = argv[2]
except:
    outname = 'out'
try:
    convtype = argv[3]
except:
    convtype = 'sift'

outpath = filepath + "_" + outname

obj = convtype == 'autoencoder' or convtype == 'capsencoder'
if obj: # if the converter uses an object, initialise it
    obj_to_use = inits[convtype]()

image_paths = []
dirpaths = []

print("Collecting paths...")
for dirpath, dirnames, filenames in walk(filepath):
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.JPEG'):
            image_paths.append(path.join(dirpath, filename)[len(filepath):])
            dirpaths.append(dirpath[len(filepath):])
print("done!")

print("Converting images to {}...".format(convtype))
workload = len(image_paths)
for i, impath in enumerate(image_paths):
    if not path.exists(outpath + dirpaths[i]):
        makedirs(outpath + dirpaths[i])

    im = imread(filepath + impath)
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    if im.shape[2] == 1: # i.e. if it's a greyscale image, convert it into a 3-channel image
        im = np.concatenate((im, im, im), axis=2)

    if obj:
        outim = refs[convtype](obj_to_use, im)
    else:
        outim = refs[convtype](im)
    savepath = outpath + impath
    np.save(savepath[:len(savepath) - 1 - savepath[::-1].index['.']] + '.npy', outim)
    stdout.write('\r{:.1%} complete'.format(i / (workload - 1)))
    stdout.flush()
stdout.write('\ndone!')