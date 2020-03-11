import torch
import numpy as np
import torchvision.transforms as transforms
import time

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd = 0.1, eigval=torch.Tensor([0.2175, 0.0188, 0.0045]), eigvec=torch.Tensor([[-0.5675,  0.7192,  0.4009],[-0.5808, -0.0045, -0.8140],[-0.5836, -0.6948,  0.4203]])):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        if self.alphastd == 0:
            return img
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return transforms.ToPILImage()(img.add(rgb.view(3, 1, 1).expand_as(img)))