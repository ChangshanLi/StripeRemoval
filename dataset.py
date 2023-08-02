import numpy as np
import os
import cv2
import random
import glob
import h5py
from scipy import ndimage
from aicsimageio import AICSImage
import torch
import torch.utils.data as udata

def image3patch(img, win, stride):
    k = 0
    img_c,img_z, img_w, img_h = img.shape
    # 按照stride的大小進行像素的擷取
    patch = img[:,:, 0:img_w-win+1:stride, 0:img_h-win+1:stride]
    total_patch_num = patch.shape[2] * patch.shape[3]
    patches = np.zeros([img_c,img_z, win*win, total_patch_num], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:,:, i:img_w-win+i+1:stride, j:img_h-win+j+1:stride]
            patches[:, :,k, :] = np.array(patch[:]).reshape(img_c,img_z, total_patch_num)
            k += 1

    return patches.reshape([img_c,img_z, win, win, total_patch_num])

def data_augmentation(image, mode):
    # 將維度轉成(w, h, c)
    output=image
    if mode == 0:
        # 不進行處理
        output = output
    elif mode == 1:
        # 上下翻轉
        output = output
        output = np.flipud(output)
    elif mode == 2:
        # 順時鐘轉180度
        output = np.rot90(output, k=2)
    elif mode == 3:
        # 順時鐘轉180度、上下翻轉
        output = np.rot90(output, k=2)
        output = np.flipud(output)
    return  output,mode
def prepare_data(data_path, patch_size, stride, aug_times=1):
    print('處理訓練資料中')
    # 訓練集路徑
    filenames = glob.glob(data_path + "/org_img" + "/*.tif")
    noise_pth = data_path + "/noise_img/"
    # 建立h5檔以進行資料的儲存
    h5_file = h5py.File('3Dtrain.h5py', 'w')
    h5_file2 = h5py.File('3Dtrain_noise.h5py', 'w')
    train_num = 0
    for file in filenames:
        org_img = AICSImage(file)
        org_img_data = org_img.data[0,0,:,:,:]
        org_img_data_max=  np.max(org_img_data)
        noise_img = AICSImage(noise_pth + file.split("\\")[-1].replace("train", "noise"))
        noise_img_data = noise_img.data[0,0,:, :, :]
        noise_img_data_max = np.max(noise_img_data)
        img_data_max= max(org_img_data_max,noise_img_data_max)
        org_img_data = org_img_data / (img_data_max)
        noise_img_data = noise_img_data/(img_data_max)
        org_img_data = np.expand_dims(org_img_data.copy(), 0)
        noise_img_data = np.expand_dims(noise_img_data.copy(), 0)
        print('处理：' + file + ' 图像')
        org_img_patches = image3patch(org_img_data, win=patch_size, stride=stride)
        noise_img_patches = image3patch(noise_img_data, win=patch_size, stride=stride)
        for n in range(org_img_patches.shape[4]):
            data = org_img_patches[:, :, :, :, n].copy()
            data2 = noise_img_patches[:, :, :,:, n].copy()
            h5_file.create_dataset(str(train_num), data=data)
            h5_file2.create_dataset(str(train_num), data=data2)
            train_num += 1
    h5_file.close()
    h5_file2.close()

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5_file = h5py.File('3Dtrain.h5py', 'r')
            h5_file2 = h5py.File('3Dtrain_noise.h5py', 'r')
        else:
            h5_file = h5py.File('3Dval.h5py', 'r')
            h5_file2 = h5py.File('3Dval_noise.h5py', 'r')
        self.keys = list(h5_file.keys())
        random.shuffle(self.keys)
        h5_file.close()
        h5_file2.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5_file = h5py.File('3Dtrain.h5py', 'r')
            h5_file2 = h5py.File('3Dtrain_noise.h5py', 'r')
        else:
            h5_file = h5py.File('3Dval.h5py', 'r')
            h5_file2 = h5py.File('3Dval_noise.h5py', 'r')
        key = self.keys[index]
        data = np.array(h5_file[key])
        data2 = np.array(h5_file2[key])
        h5_file.close()
        h5_file2.close()
        return torch.Tensor(data),torch.Tensor(data2)