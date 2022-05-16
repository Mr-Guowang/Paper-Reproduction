import torch
from torch import nn
from tqdm import tqdm
import Predict
def train(epochs,model,data,Loss,optim,device):
    for epoch in range(epochs):
        total_loss = 0
        acc_num = 0.0
        num = 0.0
        progress_bar = tqdm(data)
        for  batch in progress_bar:
            out = model(batch.src.to(device), batch.trg.to(device), batch.src_mask.to(device), batch.trg_mask.to(device))
            loss = 0
            for i in range(out.shape[0]):
                loss += Loss(out[i],batch.trg_y[i].to(device))   #因为这里损失函数对输入要求batch*class
            optim.zero_grad()
            loss.backward()
            optim.step()
            out = out.to('cpu')
            acc_num+= (out.argmax(dim=-1)==batch.trg_y).sum().cpu().item()
            num += (out.shape[0]*out.shape[1])
            acc = acc_num/num
            total_loss+=loss.to('cpu')
            progress_bar.set_description("epoch %i" % (epoch + 1))
            progress_bar.set_postfix(Loss=round(total_loss.item(), 4),Acc=acc)





