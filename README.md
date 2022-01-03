You can the single file Evaluate.py as "python Evaluate.py" to get the required .csv file.

Running order that I followed in preprocessing, model training and evaluating is:-
Sentence_Vector_Generation.py
Text_Vector_Update.py
Model_train.py
Extraction.py

These are the necessary and sufficient files that generated the best solution according to me.
I had tried other models inculding Hot encoding the span, extending the model, various optimizers and various Activating Functions, the files for which are also with me and can me sent should you require. 
I am also attaching a model which generated the best .csv according to me on the test_text.csv 

All my programming was done on python and the framework used was PyTorch.
As the text data were given in chunks and for each encrypted date there were
approximately 2000 data embedding vector of 300 dimensions Therefore, to incorporate the text
data I took the average of all the text embeddings of a particular date to form 1x300 dimensional vector for each encrypted date.
I then applied PCA to obtain the 17 dimensional vector which can represent the original 300 vector dataset most closely. Here, I obtained the number 17 by the grid search. 
I then concatenated these 17 columns to the original train according to their ID’s which is the date. I then sorted the train and test data according to the ids as given in question.

Neural Network Architecture
As the data was very expansive I used a neural network to build the models. The architecture
chosen is as follows:

378->1805->1024->512->256->128->64->1

I trained our model on the training set above mentioned. In each layer I first applied the
linear transformation, then I applied the Batch Normalization. After BatchNorm applied Drop
Out of probability 0.2 and then I applied the ​ ‘ELU’​ activation function.

As I was not given the text data for the test data therefore I also build a Generative model
which was trained by taking training data as input and obtained 17 column vectors as output.
This Generative model was used to generate 17 column vectors for the text data.The
Architecture of this is as follows:

361->722->256->128->17

In this model also I used the ​ ‘ELU’​ as activation function at every layer.
I achieved an accuracy of ​ 0.98353621 ​ on this model.

# Model-1(Descriptive NN)

I chose the criterion loss as MSE loss(Mean Squared Error Loss). The optimizer that I 
selected was Adabound with initial learning rate as 0.001 and final learning rate of 0.03. I
optimized these values at different variations and chose the values which gave the minimum
loss. After defining the loss functions, I trained the model with an epoch of 10000 and a
default batch size of 64.

# Model-2(Generative NN)

J​ust like model-1 here also I used MSE loss as criterion loss and Adabound optimizer with
initial and final learning rate as 0.001 and 0.01 respectively. After defining the loss functions, I trained the model with an epoch of 5000 and a default batch size of 64.

I was continuously training our model while running the code and was submitting it online
nut without noticing the epoch values and I didn’t save any model at that time. So, I didn’t
had models saved. So when I was asked for model I again iterated all the range of epochs
and although I was not able to generate the same model but I trained the model to higher
epochs than that was our best submissions made.
