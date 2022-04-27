import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import GET_NET
import GET_DATA
import GET_Small_net
import TRAIN
import  torch.nn.functional as F
import matplotlib.pyplot as plt
device = "cuda:0" if torch.cuda.is_available() else "cpu"
Tran_dataset = GET_DATA.get_dataset( 'C:/Users/Mr.Guo/Desktop/U-NET文章复现/DATA/Train_img',
                                     'C:/Users/Mr.Guo/Desktop/U-NET文章复现/DATA/Train_label')
Test_dataset = GET_DATA.get_dataset('C:/Users/Mr.Guo/Desktop/U-NET文章复现/DATA/Test_img',
                                    'C:/Users/Mr.Guo/Desktop/U-NET文章复现/DATA/Test_label')
# net = GET_NET.Get_net().to(device)
net = GET_Small_net.Get_net().to(device)
net.train()
Loss = nn.BCELoss()
lr = 0.001
optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

if __name__ == "__main__":
    tran_dataloader = DataLoader(Tran_dataset, batch_size=10, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(Test_dataset, batch_size=10, shuffle=True, num_workers=0)
    TRAIN.train(net,optim,10,Loss,tran_dataloader,test_dataloader,device)
