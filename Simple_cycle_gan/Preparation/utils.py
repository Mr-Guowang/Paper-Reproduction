#这个文件写了很多小工具
import torch
import random
from visdom import Visdom
import time
import sys
import numpy as np
import os



class LambdaLR():
    def __init__(self, n_epochs, start_epoch, decay_start_epoch):  #总epoch，训练开始的epoch，开始衰减的epoch
        assert ((n_epochs - decay_start_epoch) > 0) #确保在结束前开始衰减，否则报错
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.start_epoch - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    #这个类实现学习率衰减的功能，用step函数输入torch.optim.lr_scheduler

def weights_init_normal(m):           #这个类实现模型参数的初始化
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

#下面需要实现一个box类，功能如下：
#是为了训练的稳定，采用历史生成的虚假样本来更新判别器，而不是当前生成的虚假样本
#定义了一个box对象，有一个数据存储表data，大小预设为50，
#它的运转流程是这样的：数据表未填满时，每次读取的都是当前生成的虚假图像，
#当数据表填满时，随机决定 1. 在数据表中随机抽取一批数据，返回，并且用当前数据补充进来 2. 采用当前数据
class box():
    def __init__(self,max_size = 50):
        assert (max_size > 0) #必须保证容器的大小要大于0
        self.max_size = max_size
        self.data = []

    def get_and_give(self,data):
        to_return = []                         #这个data的格式为tensor【batch，3，（原图尺寸）】
        for element in data.data:                            #访问batch中的每一个图片
            element = torch.unsqueeze(element, 0)            #将3，（原图尺寸）转为1，3，（原图尺寸）
            if len(self.data) < self.max_size:             #如果box没满
                self.data.append(element)                  #将这张图片添加进去
                to_return.append(element)                  #然后在to_return列表中也添加这张图片
            else:
                if random.uniform(0,1) > 0.5:               #如果满了那几按一定的概率来执行下面的操作
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())  #在to_return列表中国随机天机一个self。data的图片
                    self.data[i] = element                  #然后将该位置替换为读取的图片
                else:
                    to_return.append(element)
        return torch.cat(to_return)

def tensor2image(tensor):                                      #由于之前做过归一化，所以我们在显示图片的时候需要将图片恢复
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))                 #如果只有一个维度，那就扩展成3维
    return image.astype(np.uint8)
#由于输入的是一个batch的所有图片，所以我们只取第一张图片来变换

#下面实现一个类，可以集成显示所有的loss或者图片之类的
class Display():
    def __init__(self,epochs, batches):
        self.viz = Visdom()
        self.epochs = epochs           #总的epoch数量
        self.batches = batches         #batch数量
        self.prev_time = time.time()   #保存初始的时间
        self.mean_period = 0
        self.all_time = 0
        self.losses = {}

    def display(self,epoch,batch,losses=None, images=None,display_batch = True):
        '''
        losses = {'loss_1':loss_1, 'loss_2': loss_2}
        images = {'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B}必须写成这种形式
        epoch,batch表示当前的epoch和batch
        '''
        #这个在模型中是放到epoch里的每个batch里进行运行的，而不是每个epoch
        #这里还可以添加其他的参数，如准确率等等,display_batch表示，是否每个batch都要显示一次
        self.mean_period = (time.time() - self.prev_time)   #计算总时间
        self.all_time += self.mean_period
        self.prev_time = time.time()                         #清空初始时间

        if display_batch:
            sys.stdout.write(
                '\rEpoch:%03d/%03d Batch:[%04d/%04d] -- ' % (epoch, self.epochs, batch, self.batches))
            for i, loss_name in enumerate(losses.keys()):  # 如果不存在这个loss，那就直接录入，如果存在，那就求和
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name].data  # .data不记录梯度
                else:
                    self.losses[loss_name] += losses[loss_name].data
                if (i + 1) == len(losses.keys()):  # i+1表示当前进行到第几个key，如果正好进行到最后一个key
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / batch))
            sys.stdout.write('Time: %2f|Epoch-time:%1f' % (self.mean_period,self.all_time))  # 打印每一次训练消耗的时间
            sys.stdout.write('\n')
            for image_name, tensor in images.items():
                self.viz.image(tensor2image(tensor.data), win=image_name, opts={'title': image_name})
            # 由于输入的是一个batch的所有图片，所以我们只取第一张图片来显示
        else:
            if batch == self.batches:
                sys.stdout.write('\rEpoch:%03d/%03d  -- ' % (epoch, self.epochs))
            for i, loss_name in enumerate(losses.keys()):  # 如果不存在这个loss，那就直接录入，如果存在，那就求和
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name].data  # .data不记录梯度
                else:
                    self.losses[loss_name] += losses[loss_name].data
                if batch == self.batches:
                    if (i + 1) == len(losses.keys()):  # i+1表示当前进行到第几个key，如果正好进行到最后一个key
                        sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / batch))
                    else:
                        sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / batch))
            if batch == self.batches:
                sys.stdout.write('Time: %2f|Epoch-time:%1f' % (self.mean_period,self.all_time))  # 打印每一次训练消耗的时间
                sys.stdout.write('\n')
                for image_name, tensor in images.items():
                    self.viz.image(tensor2image(tensor.data), win=image_name, opts={'title': image_name})

        if batch == self.batches:  # 当前epoch执行完毕
            loss_list = []
            lagend = []
            self.all_time = 0
            for loss_name, loss in self.losses.items():
                loss_list.append((loss / batch).item())
                lagend.append(loss_name)
                self.losses[loss_name] = 0.0
            if epoch==1:
                 self.viz.line(X=[epoch], Y=[loss_list],win='loss',
                                                   opts={'xlabel': 'epochs', 'ylabel': 'loss',
                                                                   'title': 'Loss',"legend":lagend})
            else:
                self.viz.line(X=[epoch], Y=[loss_list],win='loss',update='append')

class save_and_load():
    def __init__(self,model_name):
        self.model_name = model_name
    def save_epoch(self,epoch ):
        if os.path.exists('./Model/' + self.model_name + '/epoch_save/epoch.pt'):
            os.remove('./Model/' + self.model_name + '/epoch_save/epoch.pt')
            torch.save(epoch, './Model/' + self.model_name + '/epoch_save/epoch.pt')
        else:
            torch.save(epoch, './Model/' + self.model_name + '/epoch_save/epoch.pt')
    def load_epoch(self):
        if os.path.exists('./Model/' + self.model_name + '/epoch_save/epoch.pt'):
            return torch.load('./Model/' + self.model_name + '/epoch_save/epoch.pt')







