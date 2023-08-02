import numpy as np
# from skimage.measure.simple_metrics import compare_psnr

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def batch_psnr(image, imclean, data_range):
    img = image.data.cpu().numpy().astype(np.float32)
    img_clean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img.shape[0]):
        psnr += compare_psnr(img_clean[i, :, :, :, :], img[i, :, :, :, :], data_range=data_range)
    return psnr/img.shape[0]
def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])
def batch_SSIM(image,imclean):
    img = image.data.cpu().numpy().astype(np.float32)
    img_clean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM=0
    for i in range(img.shape[0]):
        SSIM += compare_psnr(img_clean[i, :, :, :, :], img[i, :, :, :, :],  multichannel=False)
    return SSIM/img.shape[0]
