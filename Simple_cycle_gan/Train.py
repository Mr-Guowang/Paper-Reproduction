import torch
from torch import nn
from Preparation import Get_Data,utils
from torch.utils.data import DataLoader

def train(parameter):
    model_name = parameter.model_name
    if model_name == 'cycle_gan':
        from Model.cycle_gan import cycle_gan
        model = cycle_gan.model(parameter)  # 这个model里有所有的网络，优化器，损失函数等等，还有一个step可以进行参数更新

    save_and_load = utils.save_and_load(model_name=model_name)  # 用来读取和储存每一次训练后的epoch

    dataset = Get_Data.dataset(file=parameter.data_name)
    dataloader = DataLoader(dataset, batch_size=parameter.batchSize, shuffle=True, num_workers=0)
    batch_nums = len(dataloader)  # 计算一共有多少个batch

    Display = utils.Display(parameter.n_epochs, batch_nums)  # 显示损失和图像


    start_epoch = save_and_load.load_epoch() if parameter.continue_epoch==1 else parameter.epoch
      # 是否从上次的epoch开始训练

    for epoch in range(start_epoch, parameter.n_epochs):
        for batch, data in enumerate(dataloader):  # 下面在每个batch里进行前向计算
            real_A = data['A']
            real_B = data['B']
            losses, images = model.step(real_A, real_B)    #训练一个batch，并且返回损失和产生的图片
            Display.display(epoch=epoch + 1, batch=batch + 1, losses=losses, images=images,
                            display_batch=parameter.display_batch)
        model.lr_step()  #更新lr
        save_and_load.save_epoch(epoch)  # 储存epoch，删除上一个epoch储存的epoch值
        model.save_models()  # 以字典的形式储存模型，并且删除上一个epoch的模型
        # print(model.optimizer_D_A.state_dict()['param_groups'][0]['lr'])  #测试一下看看学习率是否更新





