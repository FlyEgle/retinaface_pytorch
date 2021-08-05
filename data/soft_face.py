"""
!/usr/bin/python
-*- coding:utf-8 -*-

@author: jiangmingchao
@datetime: 20200922
@describe: The dataset which used for the face detection
"""
import os
import os.path
import sys
import torch
import torch.utils.data as data
import json
import urllib.request as urt
import urllib.error as Error
import cv2
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from turbojpeg import TurboJPEG


class CommonFaceDetectionDataSet(data.Dataset):
    def __init__(self, train_path, preproc=None):
        super().__init__()
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        self.trainpath = train_path
        lines = [x.strip() for x in open(self.trainpath).readlines()]
        self.image_path = []
        self.image_labels = []
        self.jpeg_decode = TurboJPEG()
        for line in lines:
            data_result = json.loads(line)
            image_url = data_result["image_url"]
            image_bbox = data_result["image_bbox"]
            self.image_path.append(image_url)
            self.image_labels.append(image_bbox)
        self.image_idx = [i for i in range(len(self.image_path))]

    def __len__(self):
        return len(self.image_path)

    # read the url image
    def _url2image(self, image_path):
        context = urt.urlopen(image_path).read()
        # print(context)
        for _ in range(10):
            try:
                image = self.jpeg_decode.decode(context)
                return image 
            except Exception as e:
                image = np.asarray(bytearray(context), dtype='uint8')
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                return image

    def __getitem__(self, index):
        for i in range(10):
            try:
                img_pth = self.image_path[index]
                img_lbl = self.image_labels[index]
                if "http" in img_pth:
                    img = self._url2image(img_pth)
                else:
                    img = cv2.imread(img_pth, cv2.IMREAD_COLOR)
                if img is not None:
                    height, width, _ = img.shape
                    # targets = np.zeros((0, 15))
                    targets = []
                    landmarks = -1
                    for idx, label in enumerate(img_lbl):
                        annoations = [0.0 for _ in list(range(15))]
                        x1, x2, x3, x4 = label["x1"], label["x2"], label["x3"], label["x4"]
                        y1, y2, y3, y4 = label["y1"], label["y2"], label["y3"], label["y4"]
                        annoations[0] = float(x1)
                        annoations[1] = float(y1)
                        annoations[2] = float(x3)
                        annoations[3] = float(y3)
                        annoations[4:] = [-1]*11
                        targets.append(annoations)
                    targets = np.array(targets)
                
                    if self.preproc is not None:
                        img, targets = self.preproc(img, targets)
                        
                    return torch.from_numpy(img), targets

                else:
                    index = random.choice(self.image_idx)
                    print("cv2 imread is None", img_pth)
                    continue
            except Error.HTTPError as e:
                index = random.choice(self.image_idx)
                print(e)
                print(img_pth)
                continue
            
            except Error.URLError as e:
                index = random.choice(self.image_idx)
                print(e)
                print(img_pth)
                continue
            
            except Exception as e:
                index = random.choice(self.image_idx)
                print(e)
                print(img_pth)
                continue


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


if __name__ == "__main__":
    data_file = "/data/remote/dataset/face_detection/train_face_detection.log"
    dataset = CommonFaceDetectionDataSet(data_file)
    print(len(dataset))
    # data, target = dataset[0]
    # print(target.shape)
    for data, target in dataset:
        print(data.shape, target.shape)