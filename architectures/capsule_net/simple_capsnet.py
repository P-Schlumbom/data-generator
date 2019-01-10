"""
Keras implementation of Sabour, Frosst & Hinton's Capsule Network (https://arxiv.org/abs/1710.09829),
based very heavily on Xifeng Guo"s implementation: https://github.com/XifengGuo/CapsNet-Keras
"""
import numpy as np

from keras import layers, models
from architectures.capsule_net.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

def create_simple_capsnet_model(inputShape, nClass, routings, caps_dim=16):
    """
    Creates a Capsule Network model.
    :param inpuShape:
    :param nClass:
    :param routings:
    :return: a Keras model
    """
    x = layers.Input(shape=inputShape)

    # layer 1: standard Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(
        x)

    # layer 2: Conv2D with 'squash' activation, reshaped to [None, num_capsule, dim_capsule]
    primaryCaps = PrimaryCap(conv1, dim_capsule=8, n_channels=64, kernel_size=9, strides=2,
                             padding='valid')  # edited to have twice the number of capsules

    # layer 3: capsule layer utilising routing algorithm.
    digitCaps = CapsuleLayer(num_capsule=nClass, dim_capsule=caps_dim, routings=routings, name='digitcaps')(
        primaryCaps)

    # layer 4: auxiliary layer replacing each capsule with its length. Supposedly not necessary for TensorFlow? Must check later.
    outCaps = Length(name='capsnet')(digitCaps)

    # ------------DECODER network starts here------------- #
    # Note: the mask hides the output of every capsule except the chosen one. Hence reconstruction is based on the pose matrix produced by the chosen capsule.
    # When training, we use Y to decide which capsules to mask.
    # During evaluation, we use the capsule that produced the longest vector to mask the capsule output.
    y = layers.Input(shape=(nClass,))
    maskedByY = Mask()([digitCaps, y])  # true label used to mask output of capsule layer for training.
    masked = Mask()(digitCaps)  # mask using the capsule with maximum length (for prediction).

    # Decoder model shared for training and prediction.
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=caps_dim * nClass))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(2048, activation='relu'))  # ADDED LAYER 1
    decoder.add(layers.Dense(np.prod(inputShape), activation='relu'))  # CHANGED SIGMOID TO RELU 3
    decoder.add(layers.Dense(np.prod(inputShape), activation='linear'))  # ADDED LAYER 2
    decoder.add(layers.Reshape(target_shape=inputShape, name='out_recon'))

    # models for training and evaluation
    train_model = models.Model([x, y], [outCaps, decoder(maskedByY)])
    #eval_model = models.Model(x, [outCaps, decoder(masked)])
    eval_model = models.Model(x, [digitCaps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(nClass, caps_dim))
    noisedDigitcaps = layers.Add()([digitCaps, noise])
    maskedNoisedY = Mask()([noisedDigitcaps, y])
    manipulateModel = models.Model([x, y, noise], decoder(maskedNoisedY))
    return train_model, eval_model, manipulateModel