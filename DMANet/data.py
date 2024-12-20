import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, gt_path):
        super(type(self), self).__init__()
        self.img_path = img_path
        self.gt_path = gt_path

        # WHU
        self.img_list = sorted(os.listdir(img_path + 'full_clear/'))
        self.to_tensor = transforms.ToTensor()

    def resize_totensor(self, img,pad_h,pad_w):
        img256 = self.to_tensor(img)
        img256=img256[:3,:,:]
        pad = (int(np.floor(pad_h/2)), int(pad_h-np.floor(pad_h/2)), int(np.floor(pad_w/2)), int(pad_w - np.floor(pad_w/2)))
        img256 = torch.nn.functional.pad(img256, pad, mode='constant', value=0)
        img256 = img256.unsqueeze(0)
        img128 = F.interpolate(img256, scale_factor=0.5, mode='bilinear')
        img64 = F.interpolate(img128, scale_factor=0.5, mode='bilinear')

        return img256.squeeze(0), img128.squeeze(0), img64.squeeze(0)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        # WHU
        blurry_img_name = os.path.join(self.img_path, 'source_1', self.img_list[idx])
        blurry_img_name2 = os.path.join(self.img_path, 'source_2', self.img_list[idx])
        clear_img_name = os.path.join(self.img_path, 'full_clear', self.img_list[idx])


        blurry_img = Image.open(blurry_img_name)
        blurry_img2 = Image.open(blurry_img_name2)
        h, w = blurry_img.size
        aim_h = int(np.ceil(h / 16) * 16)
        aim_w = int(np.ceil(w / 16) * 16)
        pad_h = int((aim_h - h) / 1)
        pad_w = int((aim_w - w) / 1)

        # WHU
        clear_img = Image.open(clear_img_name)

        assert blurry_img.size == clear_img.size

        img256, img128, img64 = self.resize_totensor(blurry_img,pad_h,pad_w)
        img2562, img1282, img642 = self.resize_totensor(blurry_img2,pad_h,pad_w)
        label256, label128, label64 = self.resize_totensor(clear_img,pad_h,pad_w)
        # label256 = self.to_tensor(clear_img)
        batch = {'img256': img256, 'img128': img128, 'img64': img64,'img2562': img2562, 'img1282': img1282, 'img642': img642, 'label256': label256, 'label128': label128,
                 'label64': label64, 'mask256': label256, 'mask128': label128,
                 'mask64': label64}
        return batch,self.img_list[idx],pad_h,pad_w

