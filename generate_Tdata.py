import numpy as np
import cv2
import copy
from aicsimageio import AICSImage
import SimpleITK as sitk
import glob
# reference brightness data生成
data_path='./data/'
filesName = glob.glob(data_path + "reference_data" + "/*.tif")
filesName.sort()
shading=0
for file in filesName:
    reader=AICSImage(file)
    data=reader.data[0,0,0,:,:]
    data_max=int(np.max(data))
    data_min=int(np.min(data))
    data_range=data_max-data_min+1
    hist = cv2.calcHist(data, [0], None, [data_range], [data_min, data_max + 1])
    hist_index=np.argmax(hist)
    data_mode=hist_index+data_min
    data[data>500]=data_mode
    data_max=np.max(data)
    shading+=cv2.GaussianBlur(data,(255,255),65,4)
template_shading=shading/10
correction_data_arr=sitk.GetImageFromArray(template_shading.astype(np.float32))
sitk.WriteImage(correction_data_arr,
                    data_path+'reference_data/template//template.tif')