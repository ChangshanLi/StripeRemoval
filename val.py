import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from utils import batch_psnr
import SimpleITK as sitk
from aicsimageio import AICSImage
from practice_models import UNet
from matplotlib import pyplot as plt


def downsampling(org_img,scales):
    target_img =Variable(torch.tensor(np.zeros(shape=(org_img.data.shape[0],org_img.data.shape[1],org_img.data.shape[2], int(org_img.data.shape[3] / scales), int(org_img.data.shape[4] / scales)), dtype=np.float32)).cuda())
    for ii in range(target_img.shape[3]):
        for jj in range(target_img.shape[3]):
            target_img[:,:,:, ii, jj] = org_img[:,:,:, scales* ii, scales * jj]
    return target_img
def interpolation(org_img,scales):
    target_img = Variable(torch.tensor(np.zeros(shape=(
    org_img.data.shape[0], org_img.data.shape[1], org_img.data.shape[2], int(org_img.data.shape[3] * scales),
    int(org_img.data.shape[4] * scales)), dtype=np.float32)).cuda())
    for ii in range(target_img.shape[3]):
        for jj in range(target_img.shape[4]):
            scrx = round((ii + 1) /(scales))
            scry = round((jj + 1) /(scales))
            target_img[:,:,:, ii, jj] = org_img[:,:,:, scrx - 1, scry - 1]
    return target_img
def main():
    # 参数设置
    logs_path = './logs/'
    data_path ='./data/val/'
    noise_pth = data_path + "/noise_img/"
    restore_pth='./data/restore/'


    # 载入模型
    print('載入模型中......\n')
    net = UNet(ch_in=1)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(logs_path, 'epoch_49net.pth')))
    model.eval()

    print("载入测试资料中......\n")
    filesName = glob.glob(data_path + "org_img" + "/*.tif")
    filesName.sort()
    for file in filesName:
        clean_img = AICSImage(file)
        clean_data = clean_img.data[0, 0, :, :, :]
        clean_data_max = np.max(clean_data)
        stripy_img = AICSImage(noise_pth + file.split("\\")[-1].replace("val", "noise"))
        stripy_data = stripy_img.data[0, 0, :, :, :]
        stripy_data_max = np.max(stripy_data)
        data_max = max(clean_data_max, stripy_data_max)
        clean_data = clean_data / (data_max)
        stripy_data = stripy_data / (data_max)
        degraded_data = clean_data + stripy_data
        degraded_data = np.expand_dims(degraded_data, 0)  # 增加batch_size，其值為1
        degraded_data = np.expand_dims(degraded_data, 1)  # 增加通道數，其值為1
        input_data = torch.Tensor(degraded_data)
        input_data = Variable(input_data).cuda()
        with torch.no_grad():
            output_data= torch.clamp(input_data+model(input_data),0.,1.)
            input_data2_downsampling = downsampling(output_data, 4)
            output_test2 = interpolation(model(input_data2_downsampling), 4)
            restore_img2 = torch.clamp(output_data+output_test2,0.,1.)
            img = restore_img2.to(torch.device("cpu"))
            img = np.float32(img.data[0, 0, :, :, :])
            newimage = sitk.GetImageFromArray(img*data_max)
            sitk.WriteImage(newimage,
                            restore_pth + file[(file.rfind("\\")+1):]+"_restore.tif")
            clean_data = np.expand_dims(clean_data.copy(), 0)
            restore_img_data = np.expand_dims(img.copy(), 0)
            clean_data = torch.Tensor(clean_data)
            restore_img_data = torch.Tensor(restore_img_data)
            clean_data = torch.unsqueeze(clean_data, 0)
            restore_img_data = torch.unsqueeze(restore_img_data, 0)
            imgn, clean_data = Variable(restore_img_data.cuda()), Variable(clean_data.cuda())
            psnr_val = batch_psnr(imgn, clean_data, 1.)
            print(psnr_val)




if __name__ == '__main__':
    main()