# load trained models and generate rconstructed images etc.
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
from PIL import Image
from architectures.capsule_net.simple_capsnet import create_simple_capsnet_model
#from architectures.capsule_net.capsnet import create_capsnet_model
from architectures.capsule_net.capsulelayers import CapsuleLayer

testpath = "testing/sphere_test-7.png"; shapeID = 3
#testpath = "testing/tetrahedron_test-1.png"; shapeID = 4
#testpath = "testing/background gradient/sphere_test-5.png"
#testpath = "testing/BFS.jpg"
#testpath = "testing/displacement-atomic-test/me-1.jpg"
#testpath = "testing/displacement-atomic-test/sphere_test-5.png"
#testpath = "testing/activation gid/originals/gid1.png"; shapeID = 1

#modelpath = "models/atomic-auto/atomic-auto_autoencoder.h5"
im = imread(testpath)
#im = np.asarray(Image.open(testpath).convert('L'))
#print(im.shape)

#im = imread(testpath, mode="RGBA")[:,:,:3]
#modelpath = "models/displacement-atomic/displacement-atomic_run1_eval.h5"
modelpath = "models/displacement-atomic_trained_capsnet_model.h5"; extra=""
#modelpathB = "models/simple-auto_autoencoder.h5"
#modelpath = "models/simple-auto_encoder.h5"
#modelpath = "models/capsnet models/cifar-testModel_trained_capsnet_model.h5"
#modelpath = "models/test-atomic_trained_capsnet_model.h5"; extra="_base"

# LOADING MODEL APPROACH
#model = load_model(modelpath)
#modelB = load_model(modelpathB)
#model = load_model(modelpath, custom_objects={"CapsuleLayer" : capsulelayers.CapsuleLayer})

# LOADING WEIGHTS APPROACH
train, eval, manip = create_simple_capsnet_model((32,32,3), 5, 3)
#train, eval, manip = create_simple_capsnet_model((32,32,3), 10, 3) # cifar dataset
train.load_weights(modelpath)

#im = imread(testpath)
if len(im.shape) == 2:
    im = np.expand_dims(im, axis=2)
if im.shape[2] == 1: # i.e. if it's a greyscale image, convert it into a 3-channel image
    im = np.concatenate((im, im, im), axis=2)

#u = 0
#im[im == 0] = u # make black grey

x = np.asarray([im])

#y = model.predict(x) # normal model
#yB = modelB.predict(x)

# converting to json
"""json_string = eval.to_json()
print(json_string)
test = model_from_json(json_string, custom_objects={'CapsuleLayer':CapsuleLayer})
test.load_weights(modelpath)"""

y = eval.predict(x)
#y = test.predict(x) # for json testing


print(y[0].shape, np.amax(y[1]), np.amin(y[1]))
plt.subplot(1,3,1)
plt.imshow(x[0])
plt.subplot(1, 3, 2)
plt.imshow(y[0][0])
plt.subplot(1, 3, 3)
plt.imshow(y[1][0].astype(int))
#plt.savefig(testpath[:-4]+'_summary.jpg', dpi=900)
plt.show()

eval.save("models/honours-save.h5")

#imsave(testpath[:-4]+'_pose-matrix.png', y[0][0].astype(int))
#imsave(testpath[:-4]+'_recon.png', y[1][0].astype(int))

"""shapeHot = np.zeros((1,5,))
shapeHot[0][shapeID] = 1
noise = np.zeros((1,5,16))
x_recons = []
scale=0.25
#[-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
recon_im = np.zeros((16*32,11*32,3))
for dim in range(16):
    for r in range(-5,6):
        tmp = np.copy(noise)
        tmp[:,shapeID,dim] = r*scale
        x_recon = manip.predict([x, shapeHot, tmp])
        x_recons.append(x_recon)
        col = r+5
        #print(recon_im.shape, recon_im[dim*32:(dim*32)+32,col*32:(col*32)+32,:].shape, dim*32, col*32)
        recon_im[dim*32:(dim*32)+32,col*32:(col*32)+32,:] = x_recon
#for i in range(len(x_recons)):
    #plt.subplot(16,11,i+1)
    #plt.imshow(x_recons[i][0].astype(int))
plt.imshow(recon_im.astype(int))
plt.savefig(testpath[:-4]+extra+'_grid-summary.jpg', dpi=900)
plt.show()

imsave(testpath[:-4]+extra+'_grid-recon.png', recon_im.astype(int))"""

#imsave(testpath[:-4]+'_recon.png', y[1][0].astype(int))

"""print(y[0].shape, np.amax(y[0]), np.amin(y[0]))
encoding = np.reshape(y[0], (len(y[0]),1))
plt.subplot(1,3,1)
plt.imshow(x[0])
plt.subplot(1, 3, 2)
plt.imshow(encoding)
#plt.imshow(y[0].astype(int))
plt.subplot(1,3,3)
plt.imshow(yB[0].astype(int))
plt.show()
print(np.mean(np.sqrt((x[0] - yB[0])**2)))

#imsave(testpath[:-4] + "_mod-{}.png".format(u), x[0].astype(int))
imsave(testpath[:-4] + "_recon.png", yB[0].astype(int))
#imsave(testpath[:-4] + "_encoding.png", (encoding/np.amax(encoding)*255).astype(int))
np.save(testpath[:-4] + "_encoding.npy", encoding)"""

"""print(im.shape)
encoded_image = np.zeros(im.shape)
for i in range(encoded_image.shape[0]//32):
    for j in range(encoded_image.shape[1]//32):
        x, y = (j * 32), (i * 32)
        #encoded_image[y:y+32, x:x+32] = model.predict(np.asarray([im[y:y+32, x:x+32]]))[0]
        encoded_image[y:y + 32, x:x + 32] = eval.predict(np.asarray([im[y:y + 32, x:x + 32]]))[1][0]

print(encoded_image.shape, np.amax(encoded_image[0]), np.amin(encoded_image[0]))
plt.subplot(2,1,1)
plt.imshow(im)
plt.subplot(2,1, 2)
plt.imshow(encoded_image.astype(int))
plt.show()"""