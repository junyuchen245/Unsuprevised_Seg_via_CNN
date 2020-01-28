from keras import backend as K
import losses, nets
import sys, os, nrrd
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, load_model
import numpy as np
from skimage.transform import rescale, resize
'''
Author: Junyu Chen
Email: jchen245@jhu.edu
'''


'''
pre-processing imgs
'''
def pre_proc(img3D, gt3D):
    for i in range(img3D.shape[0]):
        img2D = img3D[i, :, :]
        gt = gt3D[i, :, :]

        img3D[i, :, :] = (img2D - img2D.min()) / (img2D.max() - img2D.min())
        gt3D[i, :, :] = gt>0
    return img3D, gt3D

'''
Model parameters
'''
sz_x = 192
sz_y = 192
sz_z = 1
ndim = 2

'''
Initialize GPU
'''
if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    print('GPU Setup done')

'''
Initialize Models
'''
# Registration model
net = nets.recurrentNet_fine_tune((sz_x, sz_y, sz_z), ndim)#, (2, 2), (2, 2), (16, 32, 64), (32, 16))
print(net.summary())
seg_model = Model(inputs=net.inputs, outputs=net.outputs)
seg_model.load_weights('seg_model_0_0005.h5')
seg_model.compile(optimizer=Adam(lr=5e-4), loss=losses.ACWE_label().loss)

# load target image:
img = np.load('/netscratch/jchen/SPECTSeg_data/test/img41.npz')
gt3D = img['label']
img3D = img['spect']

#img, heard = nrrd.read('/netscratch/jchen/SPECTSim/xcat_r81b/patientConv1/os22/osads.nl.n1.1.nrrd')
fine_tune_num = 80
img_train = img3D[0:fine_tune_num,:,:]
gt_train = gt3D[0:fine_tune_num,:,:]
#img = img['a']
#img = img[190:190+64,178:178+64]

img_train, gt_train = pre_proc(img_train, gt_train)
img_train = img_train.reshape(fine_tune_num,sz_x, sz_y,1)
gt_train = gt_train.reshape(fine_tune_num,sz_x, sz_y,1)
for iter_i in range(801):
    loss = seg_model.train_on_batch(img_train, gt_train)
    print('loss = ' + str(loss))
    if iter_i % 100 == 0:
        seg_model.save_weights('seg_fine_tuned.h5')
        print(iter_i)
        dice_all = []
        for slice_i in range(fine_tune_num,200):
            img = img3D[slice_i,:,:]
            gt = gt3D[slice_i,:,:]
            img = (img - img.min()) / (img.max() - img.min())
            img = img.reshape(1, sz_x, sz_y, 1)
            gt = gt.reshape(1, sz_x, sz_y, 1)
            img_seg = seg_model.predict(img)
            img_seg = img_seg>0
            dice = np.sum(img_seg[gt == 1]) * 2.0 / (np.sum(img_seg) + np.sum(gt))
            dice_all.append(dice)
            if slice_i == 50:
                plt.figure(num=None, figsize=(26, 6), dpi=150, facecolor='w', edgecolor='k')
                plt.subplot(1, 3, 1)
                plt.axis('off')
                plt.imshow(img[0, :, :, 0], cmap='gray')
                plt.title('Original Image')
                plt.subplot(1, 3, 2)
                plt.axis('off')
                plt.imshow(img[0, :, :, 0], cmap='gray')
                plt.title('Image with Contour')
                imgc = img_seg[0, :, :, 0]
                plt.contour(imgc>=0.5, colors='r')
                #imgc = img_seg[0, :, :, 1]
                #plt.contour(imgc >= 0.5, colors='g')
                plt.subplot(1, 3, 3)
                plt.axis('off')
                plt.imshow(img[0, :, :, 0], cmap='gray')
                plt.title('Image with Contour')
                imgc = gt[0, :, :, 0]
                plt.contour(imgc >= 0.5, colors='r')
                plt.savefig('out_seg.png')
                plt.close()
            # sys.exit(0)
            #a = input('enter')
        print(' Dice: ' + str(np.mean(dice_all)) + ' std: ' +str(np.std(dice_all)))