from keras import backend as K
import losses, nets, random
import sys, os, nrrd
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, load_model
import numpy as np
from skimage.transform import rescale, resize
import img_flow, img_flow_test
from glob import glob

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
batch_size = 20
n_batches_per_epoch = 50

imgDir = '/netscratch/jchen/SPECTSeg_data/train/'
gtDir = '/netscratch/jchen/SPECTSeg_data/train/'
imgTestDir = '/netscratch/jchen/SPECTSeg_data/test/'
gtTestDir = '/netscratch/jchen/SPECTSeg_data/test/'

img_names = ['p1.npz', 'p2.npz', 'p3.npz']
gt_names = ['p1.npz', 'p2.npz', 'p3.npz']
img_test_names = ['t1.npz']

'''
function for loading files
'''
def load_imgs(img_dir, gt_dir, img_name, gt_name):
    img = np.load(img_dir+img_name)
    img = img['arr_0']
    gt = np.load(gt_dir+gt_name)
    gt = gt['arr_0']
    rand_idx = np.random.permutation(img.shape[0])
    img = img[rand_idx]
    gt = gt[rand_idx]
    return img, gt
'''
Initialize GPU
'''
if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
net = nets.recurrentNet((sz_x, sz_y, sz_z), ndim)#, (2, 2), (2, 2), (16, 32, 64), (32, 16))
print(net.summary())
seg_model = Model(inputs=net.inputs, outputs=net.outputs)
seg_model.compile(optimizer=Adam(lr=5e-4), loss=losses.ACWE(1.0,0.0005).loss)

'''
Making figure
'''
def show_figure(img_test3D, seg_test3D, gt_test3D):
    rand_num = random.randint(0, imgTest.shape[0])
    img_test = img_test3D[rand_num,...]; seg_test = seg_test3D[rand_num,...]; gt_test = gt_test3D[rand_num,...]
    plt.figure(num=None, figsize=(25, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(img_test, cmap='gray')
    plt.title('SPECT Image')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(img_test, cmap='gray')
    plt.contour(seg_test, colors='r')
    plt.title('Predicted Seg. Image')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(img_test, cmap='gray')
    plt.contour(gt_test, colors='r')
    plt.title('GT Image')
    plt.savefig('./outputs_seg/epoch_' + str(epoch) + '.png')
    plt.close()

'''
Testing
'''
def test_step(input_model, imgTest3D, gtTest3D, epoch):
    p_dice = []
    segOut3D = input_model.predict(imgTest3D)
    for i in range(imgTest3D.shape[0]):
        seg_test = segOut3D[i, :, : , :].reshape(sz_x,sz_y)
        seg_test = seg_test>0
        gt_test   = gtTest3D[i, :, : , :].reshape(sz_x, sz_y)
        dice = np.sum(seg_test[gt_test == 1]) * 2.0 / (np.sum(seg_test) + np.sum(gt_test))
        p_dice.append(dice)
    segOut3D = segOut3D.reshape(segOut3D.shape[0], sz_x, sz_y)
    segOut3D = segOut3D>0
    return p_dice, segOut3D


'''
Start Training 
'''
num_epoch = 161
for epoch in range(num_epoch):
    loss_all = 0
    count = 0
    for img_idx in range(len(img_names)):
        img3D, gt3D = load_imgs(imgDir, gtDir, img_names[img_idx], gt_names[img_idx])
        for idx in range(0, img3D.shape[0], batch_size):
            imgBatch = img3D[idx:idx + batch_size]
            gtBatch = gt3D[idx:idx + batch_size]
            imgBatch, gtBatch = pre_proc(imgBatch, gtBatch)
            imgBatch = imgBatch.reshape(batch_size, sz_x, sz_y, 1)
            gtBatch = gtBatch.reshape(batch_size, sz_x, sz_y, 1)
            batch_loss = seg_model.train_on_batch(imgBatch, imgBatch) # training does not require gt image
            loss_all += batch_loss
            count += 1
        del img3D
        del gt3D

    batch_loss_mean = loss_all / count
    print('Epoch num: ' + str(epoch) + ', loss: ' + str(batch_loss_mean))
    if (epoch) % 80 == 0:
        seg_model.save_weights('seg_mul.h5')
        all_dice = []
        for img_test_idx in range(len(img_test_names)):
            imgTest, gtTest = load_imgs(imgTestDir, gtTestDir, img_test_names[img_test_idx], img_test_names[img_test_idx])
            imgTest, gtTest = pre_proc(imgTest, gtTest)
            imgTest = imgTest.reshape(imgTest.shape[0], sz_x, sz_y, 1)
            gtTest = gtTest.reshape(gtTest.shape[0], sz_x, sz_y, 1)
            p_dice, segTest = test_step(seg_model, imgTest, gtTest, epoch)
            all_dice.append(p_dice)
            if img_test_idx==0:
                show_figure(imgTest, gtTest, segTest)
        print('epoch: '+str(epoch))
        print('mean dice: '+str(np.mean(all_dice)))
