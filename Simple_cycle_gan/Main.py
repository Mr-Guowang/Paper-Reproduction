import argparse
import torch
import os

import Test
import Train
Parameter = argparse.ArgumentParser()
Parameter.add_argument('--data_name',default='horse2zebra')               #数据集的名称
Parameter.add_argument('--model_name',default='cycle_gan')               #模型的名称
Parameter.add_argument('--epoch', type=int, default=0)               #从第几个epoch开始
Parameter.add_argument('--n_epochs', type=int, default=200)            #一共有几个epoch
Parameter.add_argument('--continue_model',type=int,default=0)              #是否继续使用上一次的模型训练，
Parameter.add_argument('--continue_epoch',type=int,default=0)              #是否继续使用上一次的epoch训练，
Parameter.add_argument('--batchSize', type=int, default=1)
Parameter.add_argument('--lr', type=float, default=0.0002)
Parameter.add_argument('--decay_epoch', type=int, default=0)        #从第几个epoch开始参数衰减
Parameter.add_argument('--device',default=torch.device('cuda'))
Parameter.add_argument('--input_nc', type=int, default=3)
Parameter.add_argument('--output_nc', type=int, default=3)
Parameter.add_argument('--train_test',default='test')                  #训练还是测试
Parameter.add_argument('--display_batch',type=int,default=0)                 #是否每个batch显示一次图片和损失
parameter = Parameter.parse_args()

if parameter.train_test == 'train':
    Train.train(parameter)
elif parameter.train_test == 'test':
    Test.test(parameter)












