import os
import cv2
import torch,gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tvutils
import SimpleITK as sitk

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import prepare_data, Dataset
from utils import batch_psnr
from models import UNet
import random



def main():
    # 设置参数
    num_workers = 2
    batch_size =10
    lr = 1e-3
    epochs = 50
    milestone = 10
    patch_size =128

    # 设置路径
    logs_path = './logs'

    print("正在加入资料集......")
    train_dataset = Dataset(train=True)
    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers,
                              batch_size=batch_size, shuffle=False)
    print("訓練樣本數:%d\n" % len(train_dataset))

    # 建立UNet模型
    net = UNet(ch_in=1)
    criterion = nn.MSELoss(size_average=False)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    torch.save(model, os.path.join(logs_path, 'net.pth'))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    writer = SummaryWriter(logs_path)
    step = 0
    for epoch in range(epochs):
        torch.set_grad_enabled(True)
        if epoch < milestone:
            current_lr = lr
        else:
            current_lr = lr / 10.

        # 设置学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("学习率:%f" % current_lr)
        for i, data in enumerate(train_loader, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            clean_data, stripes = data[0], data[1]
            clean_data = clean_data.to(torch.device("cpu"))
            stripes = stripes .to(torch.device("cpu"))
            degraded_data = clean_data + stripes
            clean_data,degraded_data, stripes = Variable(clean_data.cuda()),Variable(degraded_data.cuda()), Variable(stripes.cuda())
            # Residual Learning
            output_train = model(degraded_data)
            loss = criterion(output_train, (-stripes)) / (degraded_data.size()[0] ** 2 )
            loss.backward()
            optimizer.step()

            # 批次训练结果
            model.eval()
            restore_img = torch.clamp(degraded_data+output_train, 0., 1.)
            psnr_train = batch_psnr(restore_img, clean_data, 1.)
            print("[Epoch %d][%d/%d] Loss:%.4f, PSNR_train:%.4f" %
              (epoch + 1, i + 1, len(train_loader), loss.item(), psnr_train))

            # 每十个批次记录
            if step % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        model.eval()
        torch.set_grad_enabled(False)

        # 保存模型
        m="epoch_"+str(epoch)+"net.pth"
        torch.save(model.state_dict(), os.path.join(logs_path, m))




if __name__ == '__main__':
    prepare_data(data_path='./data/train', patch_size=128, stride=16, aug_times=2)
    main()
