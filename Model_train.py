#importing modules
import pandas as pd
import torch
if torch.cuda.is_available():
    import torch.cuda as t
import sys
import os
import torch.nn as nn
import random
import numpy as np
import adabound
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.utils.tensorboard import SummaryWriter

#setting environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#loading Files
train=pd.read_csv(r'/home/deepank/Downloads/BITGRIT/SoftBank_competition_Deepankur_Kansal/train_text.csv')
test=pd.read_csv(r'/home/deepank/Downloads/BITGRIT/SoftBank_competition_Deepankur_Kansal/test_text.csv')
print('Files loaded')

#Treating the data according to the model
y=train['target']
ID=test['id']
test=test.drop(['id'],axis=1)
train=train.drop(['id','target'],axis=1)

#Initialising Tensors
X_train=t.DoubleTensor(train.values)
Y_train=t.DoubleTensor(y.values).resize_((4176,1))
X_Test=t.DoubleTensor(test.values)

#criterion is taken as MSE loss according to problem statement.
criterion = torch.nn.MSELoss()

#Model: 378->1024->512->256->128->64->1 simple NN architecture consisting of activation function ELU, dropout of 0.2 and Batch Norm in every layer.
class Model_Bit(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.REG=nn.Sequential(
                nn.Linear(378,1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(512,256),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(256,128),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(64,1))               
        
    def forward(self, x):
        x = self.REG(x)
        return x
#initialising model
model=Model_Bit()

#this is to load model is that is to be trained for more epochs on the data.
if len(sys.argv)==2:
    state_dict=torch.load(sys.argv[1])
    model.load_state_dict(state_dict)   
else:
    model.train()
print('Model loaded')

#Taking the Optimizer Adabound which is much better than existing pre-defined optimizers in pytorch library.
optimizer = adabound.AdaBound(model.parameters(), lr=0.001,final_lr=0.03)

#to train on batch of 80% of data we take the array of rows.
lis=[i for i in range(0,4177)]
ls= random.sample(lis, int(4176*80/100))
ls_val=list(set(lis)-set(ls))

#writer initialization
writer = SummaryWriter('/home/deepank/runs3')
print('Starting Model Traiing')

score=1
# training the model
for epoch in range(1,10001):
    optimizer.zero_grad()
    
    y_pred = model(X_train[[ls]])
    loss = criterion(y_pred, Y_train[[ls]])
    
    y_val = model(X_train[[ls_val]])
    loss_val=criterion(y_val,Y_train[[ls_val]])
    
    if epoch % 20==0:
        print(' epoch: ', epoch,' loss: ', loss.item(),' valscore :',loss_val.item(),' value: ', np.exp(-np.sqrt(loss.item())),'value_val ',np.exp(-np.sqrt(loss_val.item())))
        #chaning the array of number of selected rows so that batch is changed.
        ls= random.sample(lis, int(4176*80/100))
        ls_val=list(set(lis)-set(ls))
    
    #writing to tensorboard
    writer.add_scalars('Losses_d', {'train_loss_d':loss.item(),'val_score_d':loss_val.item()}, epoch)
    writer.add_scalar('Loss/train_d', np.exp(-np.sqrt(loss.item())), epoch )
    writer.flush()
    
    #Backward
    loss.backward()
    optimizer.step()
    #storing the model and the file to submit to the platform of BITGRIT.
    if epoch%200==0 and np.exp(-np.sqrt(loss.item()))>0.98 and loss_val.item()<score:
        torch.save(model.state_dict(), '/home/deepank/Downloads/BITGRIT/model_norm2_'+str(epoch)+'_'+str(np.exp(-np.sqrt(loss_val.item())))+'.pth')
        Y_test=model(X_Test)
        ans=Y_test.cpu().detach().numpy()
        final=pd.DataFrame()
        final['id']=ID      
        final['target']=ans
        final.to_csv('model_norm2_'+str(epoch)+'_'+str(np.exp(-np.sqrt(loss_val.item())))+'.csv',sep=',',index=False)
        score=loss_val.item()
        print('Done')


#saving to a csv file
