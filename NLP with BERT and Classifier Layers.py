#############################################################
# Title: Binary Classification with BERT and some Classifiers
# Author: Thomas Handscomb
#############################################################

# Import libraries
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import transformers
import tensorflow
import keras

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Helpful control of display option in the Console
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Control the number of columns displayed in the console output
pd.set_option('display.max_columns', 20)
# Control the width of columns displayed in the console output
pd.set_option('display.width', 1000)

#~~~~~~~~~~~~~~~~~~~~~~~~~
## Bring in data and clean
#~~~~~~~~~~~~~~~~~~~~~~~~~
# Originally located below I have included a copy in my repo
# https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/blob/master/data/stsa.binary.train

df_Sentiment_Train_Full = pd.read_csv("C:\\Users\\Tom\\Desktop\\Python Code Example\\sentiment_dataset\\data\\train.csv"                                 
                                 , encoding = "ISO-8859-1")
# Rename column headings
colnamelist = ['Text', 'Label'] 
df_Sentiment_Train_Full.columns = colnamelist

# Take a random 20% sample of the data frame
frac = 0.05
df_Sentiment_Train = df_Sentiment_Train_Full.sample(frac=frac, replace=False, random_state=1)
df_Sentiment_Train.reset_index(drop=True, inplace = True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Prepare data set for loading into BERT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define which NLP model and pre-trained weights to use
NLP_model_class = transformers.DistilBertModel
NLP_tokenizer_class = transformers.DistilBertTokenizer
NLP_pretrained_weights = 'distilbert-base-uncased'

# Load pretrained tokenizer and model
NLP_tokenizer = NLP_tokenizer_class.from_pretrained(NLP_pretrained_weights)
NLP_model = NLP_model_class.from_pretrained(NLP_pretrained_weights)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tokenise the string names. This converts each word the text into an integer corresponding to that word in the
# BERT dictionary. The tokeniser also adds endpoint tokens of 101 at the start of the word and 102 at the end

# As an example of this process:
example_text = pd.Series(['A B C Hello'])
example_tokenized_text = example_text.apply((lambda x: NLP_tokenizer.encode(x, add_special_tokens=True)))
example_tokenized_text

# Tokenise the actual data
tokenized_text = df_Sentiment_Train['Text'].apply((lambda x: NLP_tokenizer.encode(x, add_special_tokens=True)))

# The input data for the BERT model needs to be uniform in width, i.e. all entries need
# to have the same length. To achieve this we pad the data set with 0's from the width
# of each tokenized_text value to the maximum length

# Determine the maximum length of the tokenized_text values
max_length = max([len(i) for i in tokenized_text.values])

# Create an array with each tokenised entry padded by this amount
padded_tokenized_text_array = np.array([i + [0]*(max_length-len(i)) for i in tokenized_text.values])
padded_tokenized_text_array.shape

# Define an array specifying the padded values - we use this later to distinguish the real data from
# the padded 0,1 data
padding_array = np.where(padded_tokenized_text_array != 0, 1, 0)
padding_array.shape

# The BERT model expects a PyTorch tensor as input
# Convert the padded_tokenized_text and padding arrays to PyTorch tensors (Note need to specify dtype = int)
padded_tokenized_text_tensor = torch.tensor(padded_tokenized_text_array, dtype = int)
padding_tensor = torch.tensor(padding_array)

type(padded_tokenized_text_tensor)

# We can view the evolution of a row of data
df_Sentiment_Train.loc[[0]] # Original data
tokenized_text[0] # Initial tokenised series
padded_tokenized_text_array[0] # Padded tokenised array
padded_tokenized_text_tensor[0] # Padded tokenised tensor

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Pass the processed torch tensor through the BERT model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Ensure the pytorch gradients are set to zero - by default these accumulate
with torch.no_grad():
    BERT_Output = NLP_model(padded_tokenized_text_tensor, attention_mask = padding_tensor)

# The output from the pre-trained model scoring is 3D tensor with:
#   The number of data set rows as rows
#   The max number of text words as columns
#   768 as the depth number of layers     
BERT_Output[0].shape

# Take the embedding slice of the last_hidden_states tensor as the feature variables. 
# This is the first CLS token for each sentence. The hidden state corresponding to this special token is designated 
# by the authors of BERT as an aggregate representation of the whole sentence used for classification tasks. 
# As such, when we feed in an input sentence to our model during training, the output is the length 768 hidden 
# state vector corresponding to this token

# Cut down the [CLS] token to extract embeddings of the text

features_df = pd.DataFrame(BERT_Output[0][:,0,:].numpy())
features_df.shape
features_df.index
features_df.head(5)

# Create a label DataFrame
labels_df = df_Sentiment_Train[['Label']]
labels_df.shape
labels_df.index
labels_df.head(5)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create train/test splits
train_features_df, test_features_df, train_labels_df, test_labels_df = \
train_test_split(features_df, labels_df, train_size = 0.75, random_state=40)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using a Naive model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# View the distribution of labels in the test_label_df
test_labels_df.groupby(['Label']).size()

# Therefore a naive model that predicts everything as 0 will achieve 53% accuracy
test_labels_df[test_labels_df['Label']==0].shape[0]/test_labels_df.shape[0]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using a logistic regression
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lr_clf = LogisticRegression()
lr_clf.fit(train_features_df, train_labels_df)

# Score the test set
lr_clf.score(test_features_df, test_labels_df)

# Score manually by comparing predictions to labels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Q: Where is the probability cutoff determined between, proba and 1 (guess 50% but how to control) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
log_reg_preds = pd.DataFrame(lr_clf.predict(test_features_df), index=test_features_df.index)
log_reg_preds_probs = pd.DataFrame(lr_clf.predict_proba(test_features_df), index=test_features_df.index)

# Combine
log_reg_predictions = log_reg_preds.merge(log_reg_preds_probs, left_index=True, right_index=True, how = 'inner')

log_reg_predictions.columns = ['Prediction','Prob_0', 'Prob_1']

# From the naive model
cut_off = test_labels_df[test_labels_df['Label']==0].shape[0]/test_labels_df.shape[0]

log_reg_predictions['Tuned_Prediction'] = (log_reg_predictions[['Prob_0']] < cut_off).astype(int)

log_reg_predictions.describe()

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels_df, log_reg_predictions[['Prediction']]))
print(accuracy_score(test_labels_df, log_reg_predictions[['Tuned_Prediction']]))

# Combine predictions and labels to view the errors on the testing set
# What I need here is the subset of df_Sentiment_Train that test_features corresponds to
df_Sentiment_Train_features = df_Sentiment_Train[df_Sentiment_Train.index.isin(test_features_df.index)]
df_Sentiment_Train_features.shape
test_features_df.shape

# Indexes should be the same between the following:
# and they are although different orderings
df_Sentiment_Train_features.index.sort_values().equals(test_features_df.index.sort_values())

# Merge the predictions onto the right of this set - merge on indexes
df_Sentiment_Train_features_scored = df_Sentiment_Train_features.merge(log_reg_predictions, left_index=True, right_index=True, how = 'inner')

df_Sentiment_Train_features_scored.shape
df_Sentiment_Train_features_scored.loc[[208]]

# Look at the data
df_Sentiment_Train_features_scored.to_csv("C:\\Users\\Tom\\Desktop\\Python Code Example\\test_out.csv")


# Put in an indicator flag to determine where the predictions are correct
def acc(row):
    if row['Label'] == row['Prediction']:
        val = 1
    else:
        val = 0
    return val

df_Sentiment_Train_features_scored['Accuracy'] = df_Sentiment_Train_features_scored.apply(acc, axis=1)

# Comparison of different ways of obtaining the same accuracy
lr_clf.score(test_features_df, test_labels_df)
print(accuracy_score(test_labels_df, log_reg_predictions[['Prediction']]))
print(accuracy_score(test_labels_df, log_reg_predictions[['Tuned_Prediction']]))
df_Sentiment_Train_features_scored['Accuracy'].mean() # Gives the final accuracy



#~~~~~~~~~~~~~~~~~~~~~~~
# Using a Neural Network
#~~~~~~~~~~~~~~~~~~~~~~~

# Set up a neural network with one input layer of 768 input neurons and an output layer of 2 neurons
nnmodel = Sequential()

# Add layers - defining number of input and output nodes appropriately
# Note the softmax activation function on the output layer to convert output to a probability
nnmodel.add(Dense(2, input_shape=(768,), activation='softmax', name='Output'))


# Define the Stochastic Gradient Descent optimizer with learning rate of 0.01
#optimizer = keras.optimizers.sgd(lr=0.01)

# Use the Adam optimiser
nnmodel.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

# View the network topology
print('Neural Network Model Summary: ')
print(nnmodel.summary())


# Reshape training and testing labels to a (row count, 2) array
#train_labels = np.array(train_labels)
#test_labels = np.array(test_labels)
#type(train_labels)
#train_labels.shape
#train_labels = train_labels.reshape(-1,1)
#test_labels = test_labels.reshape(-1,1)

# Convert labels via one-hot encoding
encoder = OneHotEncoder(sparse=False)

train_labels_cat = pd.DataFrame(encoder.fit_transform(train_labels).astype(int))
test_labels_cat = pd.DataFrame(encoder.fit_transform(test_labels).astype(int))
#type(train_labels_cat)
train_labels_cat.shape
test_labels_cat.shape

type(train_features)
type(train_labels_cat)

# Can pass DataFrames through the network
nnmodel_hist = nnmodel.fit(train_features, 
                           train_labels_cat, 
                           shuffle=True,
                           #validation_split=0.3, 
                           verbose=2, 
                           batch_size=10, 
                           epochs=20)

print(nnmodel_hist.history.keys())

# View the output plotted at each epoch

# Summarize history for accuracy
plt.plot(nnmodel_hist.history['accuracy'])
#plt.plot(nnmodel_hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(nnmodel_hist.history['loss'])
#plt.plot(nnmodel_hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Test the model on unseen data
type(test_features)
type(test_labels_cat)

results = nnmodel.evaluate(test_features, test_labels_cat)

#dir(nnmodel)
#nnmodel.weights

#print(results)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

# Score the testing set to validate the accuracy
# This should be a probability as the output activation function is softmax
nn_preds = pd.DataFrame(nnmodel.predict_classes(test_features), index=test_features.index)
nn_preds_probs = pd.DataFrame(nnmodel.predict_proba(test_features), index=test_features.index)


nn_preds.columns = ['NN_Prediction']
nn_preds_probs.columns = ['NN_Prob_0', 'NN_Prob_1']


# Merge the predictions onto the right of this set - merge on indexes
df_Sch_UK_Clean_features_full_scored = df_Sch_UK_Clean_features_scored.merge(nn_preds, left_index=True, right_index=True, how = 'inner').merge(nn_preds_probs, left_index=True, right_index=True, how = 'inner')

df_Sch_UK_Clean_features_full_scored.head(5)

# Sort and output the Scoring set
df_Sch_UK_Clean_features_full_scored.sort_values(by = ['FundID', 'ShareClass_Name'], inplace = True)

df_Sch_UK_Clean_features_full_scored.head(5)

df_Sch_UK_Clean_features_full_scored.to_csv("L:/My Documents/2020/NLP/LR_ScoredOutput.csv"
                 , encoding='utf-8', index=False)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using a random forest
# To do: learn random forest and xgboost in python/scikit learn
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Final comparison of classifier models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1. Naive
# 2. Logistic regression
# 3. Random forest
# 4. xgboost and/or Catboost
# 5. Neural network



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Learning code and example below
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd

sentiment_df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
sentiment_df_test = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/test.tsv', delimiter='\t', header=None)
type(sentiment_df)

sentiment_df.shape[0] + 

sentiment_df_test.shape[0]/sentiment_df.shape[0]

#dir(sentiment_df)
sentiment_df.columns
sentiment_df.head(5)

# Rename column headings
sentiment_df.columns = ['Text', 'Label'] 

# View the distribution of labels
sentiment_df['Label'].value_counts()
sentiment_df.groupby(['Label']).count()


dir(transformers)

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')


# Load pretrained tokenizer and model
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Or just make this easier to read (I guess the above is more flexible)
#tokenizer = ppb.BertTokenizer.from_pretrained('bert-base-uncased')
#model = ppb.BertModel.from_pretrained('bert-base-uncased')


# Right now, the variable model holds a pretrained distilBERT model -- a version of BERT that is smaller,
# but much faster and requiring a lot less memory.

# The tokenizer breaks the sentences into words and subwords
Subset_val = 2000
Subset_df = sentiment_df.iloc[0:Subset_val, :]

dir(tokenizer)

#tokenizer.do_basic_tokenize(Subset_df['Text'][:1])
#tokenizer.tokenize(Subset_df['Text'][:1])
#tokenized_test = Subset_df['Text'][:1].apply((lambda x: tokenizer.tokenize(x)))
#indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_test)

# The below encode looks like it first tokenises the sentance and then encodes the token to the BERT vocabulary
# i.e. converts the token words to the integers that correspond to the words in the BERT dictionary
tokenized = Subset_df['Text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
type(tokenized)

tokenized.shape

tokenized.values[1]

# Can convert tokenizes to a numpy array, rather than list of lists - however this looks the same
word_array = np.array([i for i in tokenized.values])
type(word_array)
word_array.shape

#tokenized.values[0]
#word_array[0]

# Padding puts 0's in to normalise the lengths of the lists
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded_word_array = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
padded_word_array.shape

#for i in range(1,10):
#    print (padded_word_array[i].shape)

# Define an array specifying the padded values
attention_mask = np.where(padded_word_array != 0, 1, 0)
attention_mask.shape

# Convert arrays to PyTorch tensors (Note need to specify dtype = int)
# which is what the BERT model needs for input
# input_ids = torch.tensor(padded_word_array)
input_ids = torch.tensor(padded_word_array, dtype = int)
attention_mask = torch.tensor(attention_mask)

type(input_ids)

#input_ids.shape

# Basically input_ids is a torch.tensor version of tokenized with the 0 padding included
tokenized.values[1]
input_ids[1]

# Ensure the pytorch gradients are set to zero - by default these accumulate
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask = attention_mask)

type(last_hidden_states)
dir(last_hidden_states)
last_hidden_states[0].shape
last_hidden_states[0][0,0,0]

# Take the embedding slice of the last_hidden_states tensor as the feature variables. 
# This is the first CLS token for each sentence. The hidden state corresponding to this special token is designated 
# by the authors of BERT as an aggregate representation of the whole sentence used for classification tasks. 
# As such, when we feed in an input sentence to our model during training, the output is the length 768 hidden 
# state vector corresponding to this token
features = last_hidden_states[0][:,0,:].numpy()
type(features)
features.shape
features[0]


# This feature array give the pre-trained BERT model weights on our data set. 
# View what the features values are for the first 10 elements of the features array
from scipy import stats
i = 0
while True:
    print(stats.describe(features[i]))    
    i = i + 1
    if(i > 10):
        break



# The labels are the earlier classifications
labels = Subset_df['Label']
labels.shape

# Train the model
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

train_features.shape
train_labels.shape

test_features.shape
test_labels.shape

type(test_features)
type(test_labels)

##############################
# Fine tune the model 
##############################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using a logistic regression
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

dir(lr_clf)

# Score the test set
lr_clf.score(test_features, test_labels)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Or score manually by comparing predictions to labels
# Combine predictions and labels
preds = pd.DataFrame(lr_clf.predict(test_features))

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, preds))

Combined = pd.concat([pd.DataFrame(test_labels).reset_index(drop=True), preds], axis=1)

Combined.columns = ['labels', 'preds']

# Put in an indicator flag to determine where the predictions are correct
def acc(row):
    if row['labels'] == row['preds']:
        val = 1
    else:
        val = 0
    return val

Combined['Accuracy'] = Combined.apply(acc, axis=1)
Combined['Accuracy'].mean() # Gives the final accuracy

# View the confusion matrix
Combined.groupby([0, 1]).size().unstack().fillna(0).astype('int')

(192+220)/Combined.shape[0]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using a neural net

# Set up a neural network with one input layer of 768 input neurons and an output layer of 2 neurons
nnmodel = Sequential()

# Add layers - defining number of input and output nodes appropriately
# Note the softmax activation function on the output layer to convert output to a probability
nnmodel.add(Dense(2, input_shape=(768,), activation='softmax', name='Output'))

#nnmodel.add(Dense(50, activation='softmax', name='output'))
#nnmodel.add(Dense(5, activation='softmax', name='output'))

# Define the Stochastic Gradient Descent optimizer with learning rate of 0.01
optimizer = keras.optimizers.sgd(lr=0.01)
nnmodel.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

# View the network topology
print('Neural Network Model Summary: ')
print(nnmodel.summary())


# Reshape training and testing labels to a (row count, 2) array
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
#type(train_labels)
#train_labels.shape
train_labels = train_labels.reshape(-1,1)
test_labels = test_labels.reshape(-1,1)

# Convert labels via one-hot encoding
encoder = OneHotEncoder(sparse=False)

train_labels_cat = encoder.fit_transform(train_labels).astype(int)
test_labels_cat = encoder.fit_transform(test_labels).astype(int)
#type(train_labels_cat)
train_labels_cat.shape
test_labels_cat.shape

nnmodel_hist = nnmodel.fit(train_features, 
                           train_labels_cat, 
                           shuffle=True,
                           #validation_split=0.3, 
                           verbose=2, 
                           batch_size=10, 
                           epochs=80)

print(nnmodel_hist.history.keys())

# View the output plotted at each epoch

# Summarize history for accuracy
plt.plot(nnmodel_hist.history['accuracy'])
#plt.plot(nnmodel_hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(nnmodel_hist.history['loss'])
#plt.plot(nnmodel_hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Test the model on unseen data
results = nnmodel.evaluate(test_features, test_labels_cat)

#print(results)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

# Score the testing set to validate the accuracy 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Note that a naive model could predict everything as a 1
# Accuracy would be the count of 1's over the total number of rows
sentiment_df[1].value_counts()[1]/sentiment_df.shape[0]

# Sci-kit learn also has a dummy (= random) classifier
from sklearn.dummy import DummyClassifier
clf_dummy = DummyClassifier()

clf_dummy.fit(train_features, train_labels)

dummy_preds = pd.DataFrame(clf_dummy.predict(test_features))

dummy_preds[0].value_counts()
dummy_preds[0].mean()

# Score the test set
clf_dummy.score(test_features, test_labels)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note that Google Colab offers free GPUs and TPUs!
# GPU: Graphical Processing Unit
# TPU: Tensor Processing Unit (designed by Google for Tensorflow)

device_name = tensorflow.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
else:
    print('Found GPU at: {}'.format(device_name))