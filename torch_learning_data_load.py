"""
GPUで学習を行いながら、CPUで画像を読み込んで前処理を行うloaderを作りたい
PILは昔、連続読み込みのバグで苦労したから使わない

高速化の基本は num_workersとpin_memory

とりあえずセグメンテーションをやらそうかと
学習画像はメモリに乗らないくらい大量にあるものとして、データセットクラスには画像パスのリストを引数に持たす


augmentationは albumentations ってのがめっちゃ強力そう
https://github.com/albumentations-team/albumentations
https://qiita.com/kurilab/items/b69e1be8d0224ae139ad

import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]


"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class MyDataset(Dataset):

    def __init__(self, images_path_list, labels_path_list):
        """
        :param images_path_list: 学習画像のパスリスト
        :param labels_path_list: 正解画像のパスリスト
        """
        super().__init__()

        self.images_path_list = images_path_list
        self.labels_path_list = labels_path_list

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
        # 画像読み込み
        img = self.imread(self.images_path_list[idx])
        label = self.imread(self.labels_path_list[idx])

        # BGRをRGBに
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # バッチの為に大きさを同じにしないといけない
        img = cv2.resize(img, dsize=(500, 500))
        label = cv2.resize(label, dsize=(500, 500))

        # augmentationをここで行う
        # if self.transform:
        #     sample = self.transform(sample)

        # torchでは画像のチャンネルが先に来るから位置を変える
        img = img.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        return img, label

    def imread(self, filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
        """パスに日本語が入っててもcv2が画像を読み込める用にするすごいやつ"""
        try:
            n = np.fromfile(filename, dtype)
            img = cv2.imdecode(n, flags)
            return img
        except Exception as e:
            print(e)
            return None


def get_test_path_list():
    p = r"D:\Users\ohkawa\Personal\mapillary\train\images"
    a = ["__CRyFzoDOXn6unQ6a3DnQ.jpg",
         "__d47X7jIPcnuZkob5C_DA.jpg",
         "__IoBfs3I6vB5ND-vqXK1A.jpg",
         "__KhdlKlVCeDQzVU2iyqYA.jpg",
         "__M2DBwhxBjZgQXkk5kwjQ.jpg",
         "__mcuCxPdWr9uat9bFikyg.jpg",
         "_0A_W6lEi-7W0RvVEiKkyQ.jpg",
         "_0P04ZWQtMtPMwx3lgLdWA.jpg",
         "_0R1piHytmNxzCyp5YjF8g.jpg",
         "_0ZLZEpBrN8d2YGqPYQSlA.jpg"]
    image_list = [os.path.join(p, x) for x in a]

    p = r"D:\Users\ohkawa\Personal\mapillary\train\labels"
    a = ["__CRyFzoDOXn6unQ6a3DnQ.png",
         "__d47X7jIPcnuZkob5C_DA.png",
         "__IoBfs3I6vB5ND-vqXK1A.png",
         "__KhdlKlVCeDQzVU2iyqYA.png",
         "__M2DBwhxBjZgQXkk5kwjQ.png",
         "__mcuCxPdWr9uat9bFikyg.png",
         "_0A_W6lEi-7W0RvVEiKkyQ.png",
         "_0P04ZWQtMtPMwx3lgLdWA.png",
         "_0R1piHytmNxzCyp5YjF8g.png",
         "_0ZLZEpBrN8d2YGqPYQSlA.png"]
    label_list = [os.path.join(p, x) for x in a]
    return image_list, label_list


if __name__ == "__main__":
    images_path_list, labels_path_list = get_test_path_list()

    dataset = MyDataset(images_path_list, labels_path_list)

    # datasetを分割できる generatorでseed固定できる
    # train_dataset, valid_dataset = random_split(dataset, [int(len(dataset) * 0.7), int(len(dataset) * 0.3)], generator=torch.Generator().manual_seed(42))

    # ローダーを用意　
    # drop_last=Trueはバッチサイズ端数を捨てる
    Loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True, drop_last = True)

    for image, lebel in Loader:

        print(type(image), image.shape)  # datasetではnumpyだったけどテンソルになってでてくるみたい

        # 画像に戻してみる
        image = image.permute(0, 2, 3, 1)  # 軸の交換
        image = image.numpy()
        plt.imshow(image[0])
        plt.show()








