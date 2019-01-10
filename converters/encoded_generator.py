import numpy as np
from keras.models import load_model
from scipy.misc import imread

modelpath = "models/atomic-auto/atomic-auto_encoder.h5"
MODELINPUT = 32 # dimension of models input region

class Encoder():
    def __init__(self, mpath=None):
        modelpath = "models/atomic-auto/atomic-auto_encoder.h5"
        if mpath:
            modelpath = mpath
        self.model = load_model(modelpath)

    def encode(self, im):
        x = np.asarray([im])
        return self.model.predict(x)

    def get_encoded_image(self, im):
        encoded_image = np.zeros((im.shape[0] // MODELINPUT, im.shape[1] // MODELINPUT, 80))
        for i in range(encoded_image.shape[0]):
            for j in range(encoded_image.shape[1]):
                x, y = (j * MODELINPUT), (i * MODELINPUT)
                encoded_image[i, j] = self.encode(im[y:y+MODELINPUT, x:x+MODELINPUT])
        return encoded_image