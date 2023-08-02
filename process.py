import numpy as np
import copy
import cv2
from aicsimageio import AICSImage
import SimpleITK as sitk

def computer(hist,data_min):
    hist_sum = 0
    hist_total = np.sum(hist)
    for i in range(hist.shape[0]):
        hist_sum += hist[i]
        if (hist_sum / hist_total) >= 0.95:
            return (i+data_min)

def Flat_Field_Correction(image,T_data):
    reader=AICSImage(T_data)
    template = reader.data[0, 0, 0, :, :]
    corected_data=copy.deepcopy(image)
    for i in range(image.shape[0]):
         meanVal = np.mean(image[i,:,:])
         corected_data[i,:,:] = image[i,:,:] * meanVal / template
    return corected_data

def Nolinear_Mapping(corrected_data,data_min,data_max):
    mapped_data = copy.deepcopy(corrected_data)

    # 求像素占比
    xyz = np.where(corrected_data == np.min(corrected_data))
    data_slices = corrected_data[list(xyz)[0], :, :]
    data_range = data_max - data_min + 1
    hist = cv2.calcHist(data_slices, [0], None, [data_range], [data_min, data_max + 1])
    data_percent = computer(hist, data_min)

    # 非线性映射--正变换
    mapped_data[mapped_data <= data_percent] = (corrected_data[corrected_data <= data_percent] - data_min) * 255* 0.95/ (
                data_percent - data_min)
    mapped_data[mapped_data > data_percent] = (corrected_data[corrected_data > data_percent] - data_percent) * 255* 0.05/ (
                data_max - data_percent) + 255*0.95
    return  mapped_data,data_percent

def Renolinear_Mapping(mapped_data,data_min,data_max,data_percent):
    # 非线性映射--逆变换
    remapped_data=copy.deepcopy(mapped_data)
    mapped_data_max=np.max(mapped_data)
    remapped_data[remapped_data > mapped_data_max * 0.95] = (mapped_data[mapped_data > mapped_data_max * 0.95] - mapped_data_max * 0.95) * (
                data_max - data_percent) / (mapped_data_max * 0.05) + data_percent
    remapped_data[remapped_data <= mapped_data_max * 0.95] = mapped_data[mapped_data <= mapped_data_max * 0.95] * (
                data_percent - data_min) / (mapped_data_max* 0.95) + data_min
    return remapped_data

def Padding(mapped_data,Padding):
    depth, rows, cols = mapped_data.shape

    # 填充Y方向
    padded_rows_data = np.zeros(shape=(depth,rows+Padding[1],cols))
    newrows = padded_rows_data.shape[1]
    padded_rows_data[:,int(Padding[1]/2):newrows-int(Padding[1]/2),:]=copy.deepcopy(mapped_data)
    for ii in range(int(Padding[1] / 2)):
        padded_rows_data[:,ii,:]=padded_rows_data[:,(Padding[1]-ii),:]
    for ii in range ((int(newrows-Padding[1] / 2)),newrows):
        padded_rows_data[:, ii, :] = padded_rows_data[:,2*newrows-Padding[1]-ii-1, :]


    # 填充X方向
    padded_cols_data=np.zeros(shape=(depth,newrows,cols+Padding[2]))
    newcols=padded_cols_data.shape[2]
    padded_cols_data[:,:,int(Padding[2]/2):newcols-int(Padding[2]/2)]=copy.deepcopy(padded_rows_data)
    for jj in range(int(Padding[2] / 2)):
        padded_cols_data[:,:,jj]=padded_cols_data[:,:,(Padding[2]-jj)]
    for jj in range(int(newcols-Padding[2]/2),newcols):
        padded_cols_data[:,:, jj] = padded_cols_data[:,:,2*newcols-Padding[2]-jj-1]


    # 填充Z方向
    padded_data = np.zeros(shape=(depth+Padding[0], newrows, newcols))
    newdepth = padded_data.shape[0]
    padded_data[int(Padding[0]/2):int(newdepth-Padding[0]/2),:]=copy.deepcopy(padded_cols_data)
    for zz in range(int(Padding[0] / 2)):
        # padded_data[zz, :, :] = padded_data[(Padding[0]-zz), :, :]
        padded_data[zz, :, :] = padded_data[int(Padding[0]/2), :, :]
    for zz in range(int(newdepth-Padding[0]/2),newdepth):
        # padded_data[zz, :, :] = padded_data[2*newdepth-Padding[0]-zz-1,:, :]
        padded_data[zz, :, :] = padded_data[int(newdepth-Padding[0]/2)-1 , :, :]
    return padded_data
# 分块
def Block(padded_data, patch_size, stride):
    k=0
    depth,rows, cols = padded_data.shape
    # 按照stride的大小進行像素的擷取
    patch = padded_data[:, 0:rows-patch_size[1]+1:stride[1], 0:cols-patch_size[2]+1:stride[2]]
    total_patch_num = patch.shape[1] * patch.shape[2]
    patches = np.zeros([depth, patch_size[1]*patch_size[2], total_patch_num], np.float32)

    for i in range(patch_size[1]):
        for j in range(patch_size[2]):
            patch = padded_data[:, i:rows-patch_size[1]+i+1:stride[1], j:cols-patch_size[2]+j+1:stride[2]]
            patches[:,k, :] = np.array(patch[:]).reshape(depth, total_patch_num)
            k += 1
    return patches.reshape([depth, patch_size[1], patch_size[2], total_patch_num])

# 拼接
def Cutting(slice_result_data,patch_size,stride):
    depth,rows,cols=slice_result_data.shape
    cutX_restore_img = np.zeros(shape=(depth, rows-3*stride[1], cols))
    cutX_depth,cutX_rows,cutX_cols=cutX_restore_img.shape
    for i in range(int(rows / patch_size[1])):
        if (i == 0):
            cutX_restore_img[:, :patch_size[1]-int(stride[1]/2), :] = slice_result_data[:, :patch_size[1]-int(stride[1]/2), :]
        elif (i == int(rows / patch_size[1]) - 1):
            cutX_restore_img[:, cutX_rows- patch_size[1] +int(stride[1]/2):, :] = slice_result_data[:,
                                                                             rows - patch_size[1]+ int(stride[1]/2):, :]
        else:
            cutX_restore_img[:, patch_size[1]- int(stride[1]/2) + stride[1] * (i - 1):patch_size[1]- int(stride[1]/2) + stride[1] * i, :] = slice_result_data[:,patch_size[1] * i + int(stride[1]/2):patch_size[1] * (i + 1) - int(stride[1]/2), :]
    cutY_restore_img = np.zeros(shape=(64, cutX_rows, cols-7*stride[2]))
    cutY_depth, cutY_rows, cutY_cols = cutY_restore_img.shape
    for i in range(int(cols/ patch_size[2])):
        if (i == 0):
            cutY_restore_img[:, :, :patch_size[2]-int(stride[2]/2)] = cutX_restore_img[:, :, :patch_size[2]-int(stride[2]/2)]
        elif (i == int(cols / patch_size[2])-1):
            cutY_restore_img[:, :, cutY_cols - patch_size[2] + int(stride[2]/2):] = cutX_restore_img[:, :,
                                                                             cutX_cols - patch_size[2] + int(stride[2]/2):]
        else:
            cutY_restore_img[:, :, patch_size[2]- int(stride[2]/2) + stride[2] * (i - 1):patch_size[2]- int(stride[2]/2) + stride[2] * i] = cutX_restore_img[:, :,patch_size[2] * i + int(stride[2] / 2):patch_size[2] * (i + 1) - int(stride[2] / 2)]
    cutted_slice_result_data = cutY_restore_img[:, int(stride[2]/2):cutY_rows - int(stride[2]/2), int(stride[2]/2): cutY_cols - int(stride[2]/2)]

    return cutted_slice_result_data
def Remove_slices(cutted_data,patch_size,stride):
    depth,rows,cols=cutted_data.shape
    stitcher_data=np.zeros(shape=(patch_size[0],rows,cols))
    newdepth=stitcher_data.shape[0]
    for i in range(int(depth/patch_size[0])):
        if(i==0):
            stitcher_data[:int(stride[0]/2),:,:]=cutted_data[int(patch_size[0]/2):int(patch_size[0]/2)+int(stride[0]/2),:,:]
        elif(i==int(depth/patch_size[0])-1):
            stitcher_data[newdepth-int(stride[0] / 2):newdepth, :, :] = cutted_data[depth-int(patch_size[0] / 2)-int(stride[0] / 2):depth-int(patch_size[0] / 2), :, :]
        else:
            stitcher_data[int(stride[0]/2):newdepth-int(stride[0] / 2)]=cutted_data[patch_size[0]*i+int(stride[0]/2):patch_size[0]*(i+1)-int(stride[0]/2),:,:]
    return  stitcher_data
