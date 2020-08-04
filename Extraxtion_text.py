#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 05:59:27 2019

@author: deepank
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 04:25:47 2019

@author: deepank
"""
import torch
if torch.cuda.is_available():
    import torch.cuda as t
import pandas as pd
import torch.nn as nn
import sys
import numpy as np
import adabound
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.utils.tensorboard import SummaryWriter
    
class Model_PCA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.REG_PCA=nn.Sequential(
                nn.Linear(361,722),
                nn.ELU(),
                nn.Linear(722,256),
                nn.ELU(),
                nn.Linear(256,128),
                nn.ELU(),
                nn.Linear(128,17))
        
    def forward(self, x):
        x = self.REG_PCA(x)
        return x


model_PCA=Model_PCA().cuda()
state_dict=torch.load(sys.argv[1])
model_PCA.load_state_dict(state_dict)
criterion = torch.nn.MSELoss()
train=pd.read_csv(r'/home/deepank/Downloads/BITGRIT/SoftBank_competition_Deepankur_Kansal/train_text.csv')
test=pd.read_csv(r'/home/deepank/Downloads/BITGRIT/SoftBank_competition_Deepankur_Kansal/test.csv')

#to be seen 30 or 31 or mean+std?
train['span']=train['span']/30
test['span']=test['span']/30
train=train.fillna(0)
test=test.fillna(0)

y=train['target']
x_test=test.drop(['id'],axis=1)
train=train.drop(['id','target'],axis=1)

X_train=t.DoubleTensor(train.iloc[0:4000,0:361].values)
X_train_val=t.DoubleTensor(train.iloc[4000:,0:361].values)
X_text=t.DoubleTensor(train.iloc[:,361:].values)
Y_train=t.DoubleTensor(y.iloc[0:4000].values).resize_((4000,1))
Y_train_val=t.DoubleTensor(y.iloc[4000:].values).resize_((177,1))
X_test=t.DoubleTensor(x_test.values)



U,S,V = torch.svd(torch.t(X_text))
PCA_X = torch.mm(X_text,U[:,:17])
X_train=torch.cat((X_train,PCA_X[:4000,:]),dim=1)
X_train_val=torch.cat((X_train_val,PCA_X[4000:,:]),dim=1)

final_pca=model_PCA(X_test)

'''
writer = SummaryWriter('/home/deepank/Downloads/runs')
optimizer_PCA= adabound.AdaBound(model_PCA.parameters(), lr=0.001,final_lr=0.03)
print('Starting PCA calculation')
for epoch in range(1,40001):
    optimizer_PCA.zero_grad()
    y_pca = model_PCA(X_train[:,:361])
    loss_pca = criterion(y_pca, X_train[:,361:])
    y_val_pca = model_PCA(X_train_val[:,:361])
    loss_pca_val=criterion(y_val_pca,X_train_val[:,361:])
    if epoch % 20==0:
        print(' epoch: ', epoch,' loss: ', loss_pca.item(),' valscore :',loss_pca_val.item(),' value: ', np.exp(-np.sqrt(loss_pca.item())))
    writer.add_scalars('Losses', {'train_loss':loss_pca.item(),'val_score':loss_pca_val.item()}, epoch)
    writer.add_scalar('Loss/train', np.exp(-np.sqrt(loss_pca.item())), epoch )
    loss_pca.backward()
    optimizer_PCA.step()
    if epoch%8000==0:
        torch.save(model_PCA.state_dict(), '/home/deepank/Downloads/model_PCA'+str(np.exp(-np.sqrt(loss_pca.item())))+'.pth')
final_pca=model_PCA(X_test)
'''
#saving to a csv file
ans=final_pca.cpu().detach().numpy()
final=pd.DataFrame(ans)
final.to_csv('Extra.csv',sep=',',index=False)
print('Done')