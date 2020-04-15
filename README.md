# Unsupervised Segmentation Using Convolutional Neural Network

This is a Tensorflow/Keras implementation of my paper:

@inproceedings{chen:MIDL2020,
title={Medical Image Segmentation via Unsupervised Convolutional Neural Network},
author={Junyu Chen},
booktitle={International Conference on Medical Imaging with Deep Learning -- Short Paper Track},
address={Montréal, Canada},
year={2020},
month={06-8 Jul},
url={https://openreview.net/forum?id=XrbnSCv4LU},
}

<a href="https://arxiv.org/abs/2001.10155">Chen, Junyu, et al. "Medical Image Segmentation via Unsupervised Convolutional Neural Network. " Short Paper Track Accepted, Medical Imaging with Deep Learning (MIDL), 2020.</a>



## Network Architecture:
![](https://github.com/junyuchen245/Unsuprevised_Seg_via_CNN/blob/master/pics/model.png)

## Evaluation and Results:
We evaluated four settings of the proposed algorithm on the task of bone segmentation in bone SPECT images:

* Mode 1: Unsupervised (self-supervised) training with L_ACWE.

* Mode 2: Mode 1 + fine-tuning using L_label with 10 ground truth (GT) labels.

* Mode 3: Mode 1 + fine-tuning using L_label with 80 GT labels.

* Mode 4: Training with L_ACWE + L_label.

The quantitative results can be found in the paper, and here are some qualitative results:
![](https://github.com/junyuchen245/Unsuprevised_Seg_via_CNN/blob/master/pics/seg_results.png)

### Comparing to traditional ACWE:
![](https://github.com/junyuchen245/Unsuprevised_Seg_via_CNN/blob/master/pics/example.png)


### <a href="https://junyuchen245.github.io"> About Myself</a>
