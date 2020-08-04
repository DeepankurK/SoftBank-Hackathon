# This file append sentence vectors from text data to train and text data. 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#reading files
#train=pd.read_csv(r'/home/deepank/Downloads/BITGRIT/SoftBank_competition_Deepankur_Kansal/train.csv')
test=pd.read_csv(r'./test.csv')
sent=pd.read_csv(r'./sentence_vectors.csv')

#filling missin numbers as 0
test=test.fillna(0)
sent=sent.fillna(0)
id_=sent['id']
#Raducing features from the test data to 17 to train with the original data
#The number 17 was found aftr iterating from 5 to 30 on the accuracy of the model in the model.py
sent = StandardScaler().fit_transform(sent.iloc[:,1:])
pca = PCA(n_components=17)
principalComponents = pca.fit_transform(sent)
#merging to genrate a PCA matrix
col=[str(i) for i in range(0,17)]
sent= pd.DataFrame(data = principalComponents, columns = col)
sent=pd.concat([id_,sent],axis=1)
#filling missin numbers as 0
d=pd.DataFrame(pd.np.empty((0, 18)))
d.columns=sent.columns
#getting sentence vectors for each encrypted date and storing in a single.
for i in range(0,1000):
    j=sent[sent['id']==test.loc[i,'id']]
    d=pd.concat([d,j])

#merging
result = pd.merge(test,d,on='id')

#dividing span by 30 to get a scaled value
result['span']=result['span']/30

#result.to_csv('train_text.csv',sep=',',index=False)
result.to_csv('test_text.csv',sep=',',index=False)
#After this operatio, the train data of 4177 rows changes to 4176 rows leaving one specific value duw to absence in sentence vector(text data) provided.
