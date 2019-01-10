import numpy as np
from keras.models import load_model
from architectures.capsule_net.simple_capsnet import create_capsnet_model

modelpath = "models/displacement-atomic_trained_capsnet_model.h5"
MODELINPUT = 32 # dimension of model's input region
CAPSULES = 5 # number of capsules used to represent image
CAPSULE_DIM = 16 # size of vector stored for each capsule

class CapsuleEncoder():
    def __init__(self, mpath=None):
        modelpath = "models/displacement-atomic_trained_capsnet_model.h5"
        if mpath:
            modelpath = mpath
        self.train, self.eval, self.manip = create_capsnet_model((32, 32, 3), 5, 3)
        self.train.load_weights(modelpath)

    def caps_encode(self, im):
        x = np.asarray([im])
        return self.eval.predict(x)


    def get_caps_encoded_image(self, im):
        encoded_image = np.zeros((im.shape[0] // MODELINPUT, im.shape[1] // MODELINPUT, 5, 16))
        for i in range(encoded_image.shape[0]):
            for j in range(encoded_image.shape[1]):
                x, y = (j * MODELINPUT), (i * MODELINPUT)
                #print(encoded_image[i,j].shape)
                #print(self.caps_encode(im[y:y+MODELINPUT, x:x+MODELINPUT]))
                encoded_image[i, j] = self.caps_encode(im[y:y+MODELINPUT, x:x+MODELINPUT])[0][0]
        return encoded_image