import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import  torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
def train(NET,OPTIM,N_epochs,LOSS,Train_dataloader,Test_dataloader,device):
    for e, epoch in enumerate(range(N_epochs)):
        progress_bar = tqdm(Train_dataloader)
        for i,data in enumerate(progress_bar):
            img = data[0].to(device)
            pre_label = NET(img)
            pre_label = pre_label.to('cpu')
            # pre_label = F.softmax(pre_label, dim=1)
            # loss = LOSS(pre_label[:,1:,:,:], data[1])-LOSS(pre_label[:,:1,:,:], data[1])
            loss = LOSS(pre_label,data[1])
            optim = OPTIM
            optim.zero_grad()
            loss.backward()
            optim.step()
        NET.eval()
        with torch.no_grad():
            for x, y in Test_dataloader:                 #显示一张图片看看
                img = NET(x.to(device)).to('cpu')
                # img = F.softmax(img, dim=1)
                # img = img.argmax(dim=1)
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.title('img')
                plt.imshow(x[0].permute(1, 2, 0))
                plt.subplot(1, 3, 2)
                plt.title('label')
                plt.imshow(y[0].permute(1,2,0))
                plt.subplot(1, 3, 3)
                plt.title('test')
                plt.imshow(img[0].permute(1,2,0), cmap='gray')
                plt.show()
                break
        NET.train()










