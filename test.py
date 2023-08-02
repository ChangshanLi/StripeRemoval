import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import SimpleITK as sitk
from aicsimageio import AICSImage
from models import UNet
from process import Flat_Field_Correction,Nolinear_Mapping,Renolinear_Mapping,Padding,Block,Cutting,Remove_slices

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
    data_path = './data/'
    restore_path=data_path+'test/restore//'
    T_data='.data/reference_data/template//template.tif'
    Padding_voxels=[64,256,256]
    patch_size=[64,512,512]
    stride=[32,256,256]

    # 载入模型
    print('載入模型中......\n')
    net = UNet(ch_in=1)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(logs_path, 'epoch_49net.pth')))
    model.eval()

    # 对于实际数据
    # 载入资料
    print("载入测试资料中......\n")
    filesName = glob.glob(data_path + "test" + "/*.tif")
    filesName.sort()
    for file in filesName:
        img = AICSImage(file)
        img_data = img.data[0, 0, :, :, :]
        org_depth,org_rows,org_cols=img_data.shape
        cutted_data=np.zeros(shape=(org_depth*3,org_rows,org_cols))

        # data pre-processing
        # corrected_data=Flat_Field_Correction(img_data,T_data)
        corrected_data = img_data
        data_min = int((np.min(corrected_data)))
        data_max = int(np.max(corrected_data))
        mapped_data,data_percent=Nolinear_Mapping(corrected_data,data_min,data_max)
        Padded_data=Padding(mapped_data,Padding_voxels)
        depth,rows,cols=Padded_data.shape
        Padded_data_max=np.max(Padded_data)
        Padded_data=Padded_data/Padded_data_max
        for zz in range(int((depth-patch_size[0])/stride[0])+1):
            slice_data = Padded_data[stride[0] * zz:patch_size[0] + stride[0] * zz, :, :]
            slice_result_data = np.zeros(
                shape=(patch_size[0], int(((rows - patch_size[1]) / stride[1] + 1) * patch_size[1]), int(((cols - patch_size[2]) / stride[2] + 1) * patch_size[2])))
            tile_data = Block(slice_data, patch_size, stride)

            # stripes removal
            for i in range(tile_data.shape[3]):
                input_data=tile_data[:, :, :, i]
                input_data = np.expand_dims(input_data, 0)  # 增加batch_size，其值為1
                input_data = np.expand_dims(input_data, 1)  # 增加通道數，其值為1
                input_data = torch.Tensor(input_data)
                input_data = Variable(input_data).cuda()
                with torch.no_grad():
                    input_data2 = torch.clamp(input_data + model(input_data), 0., 1.)
                    input_data2_downsampling = downsampling(input_data2, 4)
                    output_test2_interpolation = interpolation(model(input_data2_downsampling), 4)
                    restore_img2 = torch.clamp(input_data2 + output_test2_interpolation, 0., 1.2)
                    result_img2 = restore_img2.to(torch.device("cpu"))
                    result_img2 = np.float32(result_img2.data[0, 0, :, :, :] * (Padded_data_max))
                    print(zz,i)
                    slice_result_data[:, patch_size[1] * int(i / 8):patch_size[1]* (int(i / 8) + 1),
                    patch_size[2]* (i % 8):patch_size[2] * (i % 8 + 1)] = result_img2

            # data post_processing
            cutted_slice_result_data=Cutting(slice_result_data,patch_size,stride)
            cutted_data[patch_size[0] * zz:patch_size[0] * (zz + 1), :, :] = cutted_slice_result_data
        stitcher_data=Remove_slices(cutted_data,patch_size,stride)

if __name__ == '__main__':
    main()