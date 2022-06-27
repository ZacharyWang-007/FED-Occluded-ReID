import numpy as np
import torch
import random
import os
import math
from PIL import Image
import torchvision.transforms as T

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img



class RandomErasing_Background(object):
    def __init__(self, EPSILON = 0.5, root=None):
        self.EPSILON = EPSILON
        self.root = root
        self.occ_imgs = os.listdir(self.root)

        for img in self.occ_imgs:
            if not img.endswith('.jpg'):
                self.occ_imgs.remove(img)

        self.len = len(self.occ_imgs)
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img
            
        index = random.randint(0, self.len-1)

        occ_img = self.occ_imgs[index]
        occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')

        h, w = img.size()[1], img.size()[2] 
        h_, w_ = occ_img.height, occ_img.width

        ratio = h_ / w_
        if ratio > 2:
            # re_size = (random.randint(h//2, h), random.randint(w//4, w//2))
            re_size = (h, random.randint(w//4, w//2))
            function = T.Compose([
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
            occ_img = function(occ_img)
        else:
            # re_size = (random.randint(h//4, h//2), random.randint(w//2, w))
            re_size = (random.randint(h//4, h//2), w)
            function = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
            occ_img = function(occ_img)

        h_, w_ = re_size[0], re_size[1]

        index_ = random.randint(0, 3)
        # points = [(0, 0), (0, w), (h, 0), (h, w)]
        
        if index_==0:
            img[:, 0:h_, 0:w_] = occ_img
        elif index_==1:
            img[:, 0:h_, w-w_:w] = occ_img
        elif index_==2:
            img[:, h-h_:h, 0:w_] = occ_img
        else:
            img[:, h-h_:h, w-w_:w] = occ_img

        return img
