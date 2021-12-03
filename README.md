# Unsupervised Segmentation Using Convolutional Neural Network

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2001.10155-b31b1b.svg)](https://arxiv.org/abs/2001.10155)

This is a Tensorflow/Keras implementation of my paper:

<a href="https://openreview.net/forum?id=XrbnSCv4LU">Chen, Junyu, et al. "Medical Image Segmentation via Unsupervised Convolutional Neural Network. " Medical Imaging with Deep Learning (MIDL), 2020.</a>

:wave: Note that this method does ***binary*** segmentation. Please check out our new approach :point_right: (<a href="https://github.com/junyuchen245/Semi-supervised_FCM_Loss_for_Segmentation">***FCM loss***</a>) for unsupervised and semi-supervised loss functions for multi-class segmentation (PyTorch and TensorFlow).</span>


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

If you find this code is useful in your research, please consider to cite:

    @inproceedings{
    chen2020medical,
    title={Medical Image Segmentation via Unsupervised Convolutional Neural Network},
    author={Junyu Chen and Eric C. Frey},
    booktitle={Medical Imaging with Deep Learning},
    year={2020},
    url={https://openreview.net/forum?id=XrbnSCv4LU}
    }

### <a href="https://junyuchen245.github.io"> About Myself</a>
