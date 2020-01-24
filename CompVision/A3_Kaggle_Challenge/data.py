import zipfile
import os
import PIL

import numpy as np
import torchvision.transforms as transforms
import imgaug as ia
import imgaug.augmenters as iaa

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set


#data_transforms = transforms.Compose([
#    transforms.Resize((64, 64)),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#])

augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Sometimes(0.5, iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode='median'
        )),
        iaa.SomeOf((0, 5),
            [
                iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0, 1.0),
                              n_segments=(20, 200))),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
            random_order=True
        )
    ],
    random_order=True
)

class ImgAugTransform:
  def __init__(self):
    self.aug = augmenter
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)

data_transforms = {
    'train':
        transforms.Compose([
        ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        transforms.Resize(256),
        transforms.CenterCrop(224),
#        transforms.RandomResizedCrop(224),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}