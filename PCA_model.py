#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 00:44:14 2019

@author: deepank
"""
#adding MISH
import torch
import random
import pandas as pd
if torch.cuda.is_available():
    import torch.cuda as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import RAdam
import sys
import os
from datetime import datetime
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train=pd.read_csv(r'/home/deepank/Downloads/BITGRIT/train_final.csv')
test=pd.read_csv(r'/home/deepank/Downloads/BITGRIT/test_final.csv')
train=train.fillna(0)
test=test.fillna(0)
y=train['target']
ID=test['id']
test=test.drop(['id'],axis=1)
train=train.drop(['id','target'],axis=1)

from sklearn.preprocessing import StandardScaler
heads=train.columns
scaler = StandardScaler()
val=scaler.fit_transform(train)
train = pd.DataFrame(val, columns=heads)
val=scaler.transform(test)
test = pd.DataFrame(val, columns=heads)


X_train=t.DoubleTensor(train.values)
#X_train_val=t.DoubleTensor(train.iloc[4000:,:].values)
Y_train=t.DoubleTensor(y.values).resize_((4177,1))
#Y_train_val=t.DoubleTensor(y.iloc[4000:].values).resize_((177,1))
Test=t.DoubleTensor(test.values)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x=x*(torch.tanh(F.softplus(x)))
        return x


class Model_Bit(nn.Module):
    def __init__(self):
        super().__init__()
        self.REG=nn.Sequential(
                nn.Linear(60,512),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.2),
                Mish(),
                nn.Linear(512,256),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                Mish(),
                nn.Linear(256,128),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.2),
                Mish(),
                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.Dropout(p=0.2),
                Mish(),
                nn.Linear(64,16),
                nn.BatchNorm1d(16),
                Mish(),
                nn.Dropout(0.2),
                nn.Linear(16,1),
                Mish())
        self.REG2=nn.Sequential(
                nn.BatchNorm1d(9),
                nn.Linear(9,1),
                Mish())
        self.REG_text=nn.Sequential(
                nn.Linear(300,512),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.2),
                Mish(),
                #nn.Linear(512,256),
                #nn.BatchNorm1d(256),
                #nn.Dropout(p=0.2),
                #Mish(),
                nn.Linear(512,256),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                Mish(),
                nn.Linear(256,64),
                nn.BatchNorm1d(64),
                nn.Dropout(p=0.2),
                Mish(),
                nn.Linear(64,16),
                nn.BatchNorm1d(16),
                Mish(),
                nn.Dropout(0.2),
                nn.Linear(16,1),
                Mish())
        self.SPAN=nn.Sequential(
                nn.Linear(30,8),
                nn.BatchNorm1d(8),
                Mish(),
                nn.Dropout(0.2),
                nn.Linear(8,1),
                Mish())
    def forward(self, x):
        x = torch.cat((x[:,0:1],self.REG(x[:,1:61]),self.REG(x[:,61:121]),self.REG(x[:,121:181]),self.REG(x[:,181:241]),self.REG(x[:,241:301]),self.REG(x[:,301:361]),self.REG_text(x[:,361:661]),self.SPAN(x[:,661:691])),1)
        x=self.REG2(x)
        return x
    
criterion = torch.nn.MSELoss()

model=Model_Bit().cuda()
if len(sys.argv)==2:
    state_dict=torch.load(sys.argv[1])
    model.load_state_dict(state_dict)   
else:
    model.train()
lr=0.01    
optimizer = RAdam.RAdam(model.parameters(),lr=lr)


lis=[i for i in range(0,4177)]
ls= random.sample(lis, int(4177*70/100))
ls_val=list(set(lis)-set(ls))

writer = SummaryWriter('/home/deepank/runs3')
print('Starting Model Traiing')
score_val=1
score=1
for epoch in range(1,30001):
    optimizer.zero_grad()
    y_pred = model(X_train[[ls]])
    loss = criterion(y_pred, Y_train[[ls]])
    y_val = model(X_train[[ls_val]])
    loss_val=criterion(y_val,Y_train[[ls_val]])
    if epoch % 20==0:
        print(' epoch: ', epoch,' loss: ', loss.item(),' valscore :',loss_val.item(),' value: ', np.exp(-np.sqrt(loss.item())),'value_val ',np.exp(-np.sqrt(loss_val.item())))
        ls= random.sample(lis, int(4177*70/100))
        ls_val=list(set(lis)-set(ls))
    
    writer.add_scalars('Losses_d', {'train_loss_d':loss.item(),'val_score_d':loss_val.item()}, epoch)
    writer.add_scalar('Loss/train_d', np.exp(-np.sqrt(loss.item())), epoch )
    writer.flush()
    loss.backward()
    if epoch%2000==0:
        lr=lr/10
        optimizer = RAdam.RAdam(model.parameters(),lr=lr)
    optimizer.step()
    if  loss.item()<score and np.exp(-np.sqrt(loss_val.item()))>=0.98 and np.exp(-np.sqrt(loss.item()))>0.98 and loss_val.item()<score_val:
        torch.save(model.state_dict(), '/home/deepank/Downloads/BITGRIT/model_norm2_'+str(epoch)+'_'+str(np.exp(-np.sqrt(loss_val.item())))+'.pth')
        Y_test=model(Test)
        ans=Y_test.cpu().detach().numpy()
        final=pd.DataFrame()
        final['id']=ID      
        final['target']=ans
        now = datetime.now()
        current_time = now.strftime("%d_%H_%M_%S")
        final.to_csv('model_norm2_'+str(epoch)+'_'+str(np.exp(-np.sqrt(loss_val.item())))+'.csv',sep=',',index=False)
        score_val=loss_val.item()
        score=loss.item()
        print('Done')


#saving to a csv file
