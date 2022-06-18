import glob
import os
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class dataset(Dataset):
    def __init__(self,root ='./Preparation/Dataset',file = 'horse2zebra',train_test = 'train',transform = None,unaligned=False):
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, file, '%sA' % train_test) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, file, '%sB' % train_test) + '/*.*'))
        # 其中 * 表示匹配任意字符，如. / Dataset / horse2zebra / trainA / n02381460_1001.jpg
        # 第一个 * 匹配了n02381460_1001，第二个 * 匹配了jpg
        if transform is None:
            compose = [
                transforms.ToTensor(),  # 这里会有一个归一化torch.Size([3, 256, 256])
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 这里就是减去均值再除以方差(标准化）
            ]
            self.transform = transforms.Compose(compose)
        else:
            self.transform = transform

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        '''
        x = [1, 2, 3, 4, 5, 6, 7]
        for i in range(len(x)):
            print(x[i % len(x)])    #输出 1,2,3,4,5,6,7
        '''
        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        if image_A.mode != "RGB":                  #作者提供的数据集里由灰度图像，需要转化为RGB
            image_A = image_A.convert('RGB')
        if image_B.mode != "RGB":
            image_B = image_B.convert('RGB')

        image_A = self.transform(image_A)
        image_B = self.transform(image_B)

        return {'A': image_A, 'B': image_B}

    def __len__(self):

        return max(len(self.files_A), len(self.files_B))




'''
x = [1,2,3,4,5,6,7,8,9]
y = [1,2,3,4]
class data(Dataset):
    def __init__(self,x,y):
        self.x= x
        self.y = y
    def __len__(self):
        return len(self.x)-0
    def __getitem__(self, item):
        x1 =  x[item%len(self.x)]
        y1 =  y[item%len(self.y)]
        return {'x':x1,'y':y1}
dadad = data(x,y)
z = 0
for i in dadad:
    print(i)
    z += 1
    if z==11:
        break
train_iter = torch.utils.data.DataLoader(dadad,batch_size = 1,shuffle=True)
for i in train_iter:
    print(i)
以上代码，dataset里其实会一直循环，在整个列表【index】不停的循环
而dataloader会去len（dataset）的数量的值进行迭代
所以运行结果如下
{'x': 1, 'y': 1}
{'x': 2, 'y': 2}
{'x': 3, 'y': 3}
{'x': 4, 'y': 4}
{'x': 5, 'y': 1}
{'x': 6, 'y': 2}
{'x': 7, 'y': 3}
{'x': 8, 'y': 4}
{'x': 9, 'y': 1}
{'x': 1, 'y': 2}
{'x': 2, 'y': 3}
{'x': tensor([2]), 'y': tensor([2])}
{'x': tensor([7]), 'y': tensor([3])}
{'x': tensor([4]), 'y': tensor([4])}
{'x': tensor([9]), 'y': tensor([1])}
{'x': tensor([6]), 'y': tensor([2])}
{'x': tensor([8]), 'y': tensor([4])}
{'x': tensor([5]), 'y': tensor([1])}
{'x': tensor([3]), 'y': tensor([3])}
{'x': tensor([1]), 'y': tensor([1])}
'''



