import os
import torch
from Preparation import Get_Data
from torch.utils.data import DataLoader

def test(parameter):
    model_name = parameter.model_name
    models = torch.load('./Model/'+model_name+'/model_save/models.pt')
    dataset = Get_Data.dataset(file=parameter.data_name,train_test=parameter.train_test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model_name = parameter.model_name
    if model_name == 'cycle_gan':
        from Model.cycle_gan import cycle_gan

    for batch,image in enumerate(dataloader):
        cycle_gan.model(parameter).test(batch,image['A'],image['B'])






