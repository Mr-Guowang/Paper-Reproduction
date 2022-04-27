#使用了kaggle上的数据集  https://www.kaggle.com/competitions/carvana-image-masking-challenge/data
#图片比较大
#首先划分数据集，原文件图片很多，而且图片太大了，因此需要进行预处理
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import glob
Path_img = 'C:/Users/Mr.Guo/Desktop/实验数据/UNET/train'
Path_label = 'C:/Users/Mr.Guo/Desktop/实验数据/UNET/train_masks'
path_tran_img = 'C:/Users/Mr.Guo/Desktop/U-NET文章复现/DATA/Train_img'
path_tran_label = 'C:/Users/Mr.Guo/Desktop/U-NET文章复现/DATA/Train_label'
path_test_img = 'C:/Users/Mr.Guo/Desktop/U-NET文章复现/DATA/Test_img'
path_test_label = 'C:/Users/Mr.Guo/Desktop/U-NET文章复现/DATA/Test_label'
def set_image(Path_img,Path_label,path_tran_img,path_tran_label,path_test_img,path_test_label):
    img_file = os.listdir(Path_img)
    label_file = os.listdir(Path_label)
    for i, file in enumerate(img_file):
        if (i < 700):  # 读取图片
            shutil.copy(os.path.join(Path_img, file), path_tran_img)  # 将文件复制到新的文件夹中
        elif(i<1000):
            shutil.copy(os.path.join(Path_img, file), path_test_img)  # 将文件复制到新的文件夹中
    for i, file in enumerate(label_file):
        if (i < 700):  # 读取图片
            shutil.copy(os.path.join(Path_label, file), path_tran_label)  # 将文件复制到新的文件夹中
        elif(i<1000):
            shutil.copy(os.path.join(Path_label, file), path_test_label)  # 将文件复制到新的文件夹中
# set_image(Path_img,Path_label,path_tran_img,path_tran_label,path_test_img,path_test_label)
class CrypkoDataset(Dataset):                     #继承dataset类来进行图片读取
    def __init__(self, path_img,path_label, transform_img=None,transform_label=None):
        self.path_img = path_img
        self.path_label = path_label
        self.num_samples = len(self.path_img)
        if (transform_img==None):
            self.transform_img = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # convert  to tensor
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalization
                ]
            )
        else:
            self.transform_img = transform_img
        if (transform_label==None):
            self.transform_label = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),  # 改为灰度图片
                    transforms.ToPILImage(),
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # convert  to tensor
                ]
            )
        else:
            self.transform_label = transform_label

    def __getitem__(self,idx):
        Path_img = self.path_img[idx]
        Path_label = self.path_label[idx]
        img = torchvision.io.read_image(Path_img)
        label = torchvision.io.read_image(Path_label)
        img = self.transform_img(img)
        label = self.transform_label(label)
        label = torch.where(label>0,torch.ones((1,224,224)),label)  #将标签转化为2值tensor
        return img,label                                    #组合起来返回

    def __len__(self):
        return self.num_samples
def get_dataset(path_img,path_label, transform_img=None,transform_label=None):
    img_fnames = glob.glob(os.path.join(path_img, '*'))                    #获取指定文件下的所有图片的路径
    label_fnames = glob.glob(os.path.join(path_label, '*'))
    dataset = CrypkoDataset(img_fnames,label_fnames, transform_img,transform_label)
    return dataset


