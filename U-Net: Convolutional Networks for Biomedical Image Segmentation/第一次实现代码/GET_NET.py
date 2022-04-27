import torch
from torch import nn
import random
class Cov_block(nn.Module):                 #定义卷积模块
    def __init__(self,in_chanels,med_chanels,out_chanels):
        super(Cov_block,self).__init__()
        self.Cov = nn.Sequential(
            nn.Conv2d(in_chanels, med_chanels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(med_chanels),
            nn.ReLU(),
            nn.Conv2d(med_chanels, out_chanels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_chanels),
            nn.ReLU()
        )
    def forward(self,x):
        y = self.Cov(x)
        return y
def Copy_Crop(feature1,feature2):    #这里无论feature1和feature2多大，最后都会按照feature输出，而且是在feature1里随机crop
    in_size = feature1.shape[3]            #求输入尺寸
    out_size = feature2.shape[3]          #求输出尺寸
    len = int(in_size - out_size)
    First_edge = random.randint(0,len)
    Second_edge = random.randint(0,len)
    out_feature = feature1[:,:,First_edge:First_edge+out_size,Second_edge:Second_edge+out_size]#在feature1裁剪出
    return torch.cat([out_feature,feature2],dim=1)         #拼接
'''x = torch.rand((1,2,3,3))
   y = torch.rand((1,1,2,2))
   print(x.shape)
   print(y.shape)
   print(Copy_Crop(x,y).shape)'''
class Down_Sample(nn.Module):
    def __init__(self):
        super(Down_Sample, self).__init__()
        self.block = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        y = self.block(x)
        return y
class Up_Sample(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(Up_Sample, self).__init__()
        self.block = nn.ConvTranspose2d(in_dim, out_dim, 2, 2,padding=0, output_padding=0, bias=False)
        #这里使用了反卷积，也可以使用上采样，上采样的代码如下
        #nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
    def forward(self,x):
        y = self.block(x)
        return y
class Get_net(nn.Module):
    def __init__(self):
        super(Get_net, self).__init__()
        self.Down_Conv1 =  Cov_block(3,64,64)
        self.Down_Conv2 = Cov_block(64, 128, 128)
        self.Down_Conv3 = Cov_block(128, 256, 256)
        self.Down_Conv4 = Cov_block(256, 512, 512)
        self.Down_Conv5 = Cov_block(512, 1024, 1024)
        self.Down_Sample = Down_Sample()
        self.Up_Sample1 = Up_Sample(1024,512)
        self.Up_Conv1 = Cov_block(1024, 512, 512)
        self.Up_Sample2 = Up_Sample(512, 256)
        self.Up_Conv2 = Cov_block(512, 256, 256)
        self.Up_Sample3 = Up_Sample(256, 128)
        self.Up_Conv3 = Cov_block(256, 128, 128)
        self.Up_Sample4 = Up_Sample(128, 64)
        self.Up_Conv4 = Cov_block(128, 64, 64)
        self.One_point_One = nn.Conv2d(64,2,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        y1 = self.Down_Conv1(x)
        y2 = self.Down_Sample(y1)
        y2 = self.Down_Conv2(y2)
        y3 = self.Down_Sample(y2)
        y3 = self.Down_Conv3(y3)
        y4 = self.Down_Sample(y3)
        y4 = self.Down_Conv4(y4)
        y5 = self.Down_Sample(y4)
        y5 = self.Down_Conv5(y5)
        out = self.Up_Sample1(y5)
        out = self.Up_Conv1(Copy_Crop(y4,out))
        out = self.Up_Sample2(out)
        out = self.Up_Conv2(Copy_Crop(y3, out))
        out = self.Up_Sample3(out)
        out = self.Up_Conv3(Copy_Crop(y2, out))
        out = self.Up_Sample4(out)
        out = self.Up_Conv4(Copy_Crop(y1, out))
        out = self.One_point_One(out)
        return out
# x = torch.rand((3,1,572,572))
# net = Get_net()
# y = net(x)
# print(y.shape)   输出结果torch.Size([3, 2, 388, 388])
