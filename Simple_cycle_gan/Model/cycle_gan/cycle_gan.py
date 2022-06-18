import torch
from torch import nn
import torch.nn.functional as F
import os
from Preparation.utils import LambdaLR,box
import torchvision
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        #对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充
                        nn.Conv2d(in_features, in_features, 3),
                        #镜像翻转后大小+2，卷积后大小-2，所以最终输出尺寸不变
                        nn.InstanceNorm2d(in_features),
                        #Batch Normalization是指batchsize图片中的每一张图片的同一个通道一起进行Normalization操作。
                        # 而Instance Normalization是指单张图片的单个通道单独进行Noramlization操作。
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):  #输入通道，输出通道，9个res模块
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),#镜像翻转后大小+6，卷积后大小-6，所以最终输出尺寸不变
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2  #128
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),#图片尺寸缩小为一半
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2  #最终尺寸变为原来的四分之一，通道变为256 然后in_features=256 out_features变为512

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]   #尺寸和通道不变

        # Upsampling
        out_features = in_features//2    #in_features=256 out_features变为128
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        #反卷积，尺寸翻倍
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2 #最终尺寸变为原图的大小，通道变为64 然后in_features=64 out_features变为32

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]      #最终尺寸和通道都变成原图

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another                      #batch*3*256*256
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),     #batch*64*128*128
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),           #batch*128*64*64
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),          #batch*256*32*32
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),                    #batch*512*31*21
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]                              #batch*1*30*30

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten          x.size()[2:]---[30,30]
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)        #batch*1


class model():             #这个类集成了 生成器，判别器，优化器，损失函数，训练过程，测试过程，每个模型都这么写，训练过程的编写可以实现统一化
    def __init__(self,opt):
        self.device = opt.device
        self.model_name = opt.model_name

        if opt.continue_model==1:   #如果使用上一次的模型，那就读取文件夹中的模型
            models = torch.load('./Model/' + self.model_name + '/model_save/' + 'models.pt')
            self.netG_A2B = models['netG_A2B'].to(self.device)
            self.netG_B2A = models['netG_B2A'].to(self.device)
            self.netD_A = models['netD_A'].to(self.device)
            self.netD_B = models['netD_B'].to(self.device)
        elif opt.continue_model==0:
            self.netG_A2B = Generator(input_nc=opt.input_nc, output_nc=opt.output_nc).to(self.device)
            self.netG_B2A = Generator(input_nc=opt.input_nc, output_nc=opt.output_nc).to(self.device)

            self.netD_A = Discriminator(input_nc=opt.input_nc).to(self.device)
            self.netD_B = Discriminator(input_nc=opt.input_nc).to(self.device)

        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        self.optimizer_A2B = torch.optim.Adam(self.netG_A2B.parameters(),lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_B2A = torch.optim.Adam(self.netG_B2A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        self.lr_scheduler_A2B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_A2B, lr_lambda=LambdaLR(opt.n_epochs,
                                                                       opt.epoch,opt.decay_epoch).step)
        self.lr_scheduler_B2A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_B2A, lr_lambda=LambdaLR(opt.n_epochs,
                                                                       opt.epoch,opt.decay_epoch).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs,
                                                                       opt.epoch, opt.decay_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs,
                                                                       opt.epoch,opt.decay_epoch).step)
        self.fake_A_box = box()
        self.fake_B_box = box()



    def step(self,real_A,real_B):                  #进行一个batch的step
        # =======================训练判别器+++++++++++++++++++++++++++
        self.netG_A2B.to(self.device)
        self.netG_B2A.to(self.device)

        self.netD_A.to(self.device)
        self.netD_B.to(self.device)

        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        self.optimizer_A2B.zero_grad()
        self.optimizer_B2A.zero_grad()

        same_A = self.netG_B2A(real_A)          #这里计算的是identity loss，就是说B2A的生成网络不能对A有影响，训练的是生成器
        same_B = self.netG_A2B(real_B)
        loss_identity_A = self.criterion_identity(same_A, real_A) * 5.0
        loss_identity_B = self.criterion_identity(same_B, real_B) * 5.0

        fake_B = self.netG_A2B(real_A)
        pred_fake_B = self.netD_B(fake_B)
        fake_A = self.netG_B2A(real_B)
        pred_fake_A = self.netD_A(fake_A)

        target_real = torch.ones_like(pred_fake_A)  #生成标签
        target_fake = torch.zeros_like(pred_fake_A)

        loss_GAN_A2B = self.criterion_GAN(pred_fake_B,target_real)    #这里还是训练生成器，我们需要生成器骗过判别器
        loss_GAN_B2A = self.criterion_GAN(pred_fake_A, target_real)   #所以给它附真实的标签

        recovered_A = self.netG_B2A(fake_B)        #这里计算cycle_loss，我们希望A2B2A的图片和原图十分接近
        loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = self.netG_A2B(fake_A)
        loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * 10.0

        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        self.optimizer_A2B.step()
        self.optimizer_B2A.step()

        #=======================训练判别器A+++++++++++++++++++++++++++
        self.optimizer_D_A.zero_grad()
        pred_real_A = self.netD_A(real_A)
        loss_D_real_A = self.criterion_GAN(pred_real_A, target_real)   #这个我们希望他判别出真实的图片

        fake_box_A = self.fake_A_box.get_and_give(fake_A)
        pred_fake_box_A = self.netD_A(fake_A.detach())
        loss_D_fake_A = self.criterion_GAN(pred_fake_box_A, target_fake) #这里希望判别出假的图片，所以给假标签

        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # =======================训练判别器B+++++++++++++++++++++++++++

        self.optimizer_D_B.zero_grad()
        pred_real_B = self.netD_B(real_B)
        loss_D_real_B = self.criterion_GAN(pred_real_B, target_real)  # 这个我们希望他判别出真实的图片

        fake_box_B = self.fake_B_box.get_and_give(fake_B)
        pred_fake_box_B = self.netD_B(fake_B.detach())
        loss_D_fake_B = self.criterion_GAN(pred_fake_box_B, target_fake)  # 这里希望判别出假的图片，所以给假标签

        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        self.optimizer_D_B.step()

        losses = {'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
         'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
         'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}

        images = {'real_A': real_A, 'fake_A2B': fake_B,'rec_A2B2A':recovered_A,'idt_A':same_A,
                  'real_B': real_B, 'fake_B2A': fake_A, 'rec_B2A2B': recovered_B, 'idt_B': same_B}

        return losses,images  #这个函数将会进行一个batch的训练，然后返回loss和image，并且更新初始化的各个参数

    def lr_step(self):  #更新lr的参数
        self.lr_scheduler_A2B.step()
        self.lr_scheduler_B2A.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()

    def save_models(self):
        models = {
            'netG_A2B':self.netG_A2B.to('cpu'),'netG_B2A':self.netG_B2A.to('cpu'),  #先放到cpu上再保存
            'netD_A':self.netD_A.to('cpu'),'netD_B':self.netD_B.to('cpu')
        }
        # 这里由于以后可能会有很多不同的模型，所以我决定将model写成字典的形式
        # model={model1:net1,model2:net2},这里的model name意思还是文件夹的名字，也就是选用的哪个模型
        if os.path.exists('./Model/' + self.model_name + '/model_save/' + 'models.pt'): #如果模型已经存在，那就删除上一个模型
            os.remove('./Model/' + self.model_name + '/model_save/' + 'models.pt')
            torch.save(models, './Model/' + self.model_name + '/model_save/' + 'models.pt')
        else:
            torch.save(models,'./Model/' + self.model_name + '/model_save/' + 'models.pt')

    def test(self,batch,realA,realB):#这个function用来实现测试功能，将A2B，B2A的原图和生成图片拼接起来保存
        models = torch.load('./Model/' + self.model_name + '/model_save/' + 'models.pt')
        realA = realA.to(self.device)
        realB = realB.to(self.device)

        netG_A2B = models['netG_A2B'].to(self.device)
        netG_B2A = models['netG_B2A'].to(self.device)

        fake_B = netG_A2B(realA).cpu()
        fake_A = netG_B2A(realB).cpu()

        image_A2B = (torch.cat((realA.cpu(),fake_B),dim=-1)+1.0)/2
        image_B2A = (torch.cat((realB.cpu(), fake_A), dim=-1)+1.0)/2

        if os.path.exists('./results_save/'+self.model_name) == False:
            os.mkdir('./results_save/'+self.model_name)

        if os.path.exists('./results_save/'+self.model_name+'/A2B') == False:
            os.mkdir('./results_save/'+self.model_name+'/A2B')
        if os.path.exists('./results_save/'+self.model_name+'/B2A') == False:
            os.mkdir('./results_save/'+self.model_name+'/B2A')

        torchvision.utils.save_image(image_A2B.detach(),'./results_save/'+self.model_name+'/A2B/'+str(batch+1)
                                     +'.jpg', nrow=1)
        torchvision.utils.save_image(image_B2A.detach(), './results_save/' + self.model_name + '/B2A/' + str(batch + 1)
                                     +'.jpg', nrow=1)








