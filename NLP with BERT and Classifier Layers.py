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

# Take a random sample of the data frame to speed up processing in this example
frac = 0.10
df_Sentiment_Train = df_Sentiment_Train_Full.sample(frac=frac, replace=False, random_state=1)
df_Sentiment_Train.reset_index(drop=True, inplace = True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Prepare data set for loading into BERT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define which NLP model and pre-trained weights to use. In this example I use the light weight version of
# the full BERT model, called DistilBert, to speed up run time
NLP_model_class = transformers.DistilBertModel
NLP_tokenizer_class = transformers.DistilBertTokenizer
NLP_pretrained_weights = 'distilbert-base-uncased'

# Load pretrained tokenizer and model
NLP_tokenizer = NLP_tokenizer_class.from_pretrained(NLP_pretrained_weights)
NLP_model = NLP_model_class.from_pretrained(NLP_pretrained_weights)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tokenise the string names. This converts each word in the text into an integer corresponding to that word in the
# BERT dictionary. The tokeniser also adds endpoint tokens of 101 at the start of the word and 102 at the end

# As an example of this process:
example_text = pd.Series(['A B C Hello'])
example_tokenized_text = example_text.apply((lambda x: NLP_tokenizer.encode(x, add_special_tokens=True)))
example_tokenized_text

# Tokenise the actual data
tokenized_text = df_Sentiment_Train['Text'].apply((lambda x: NLP_tokenizer.encode(x, add_special_tokens=True)))

# The input data for the BERT model needs to be uniform in width, i.e. all entries need
# to have the same length. To achieve this we pad the data set with 0's from the width
# of each tokenized_text value to the maximum tokenised length in the series 

# Determine the maximum length of the tokenized_text values
max_length = max([len(i) for i in tokenized_text.values])

# Create an array with each tokenised entry padded by 0's to the max length
padded_tokenized_text_array = np.array([i + [0]*(max_length-len(i)) for i in tokenized_text.values])
padded_tokenized_text_array.shape

# Define an array specifying the padded values - we use this later to distinguish the real data from
# the padded [0] data
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
    DistilBERT_Output = NLP_model(padded_tokenized_text_tensor, attention_mask = padding_tensor)

# The output from the pre-trained DistilBERT model scoring is a 3D tensor with:
#   The number of data set rows as rows
#   The max number of text words as columns
#   768 as the depth number of layers     
type(DistilBERT_Output[0])
DistilBERT_Output[0].shape

# Slice down the [CLS] token to extract embeddings of the text. These will form the feature variables into
# the final classifier model
features_df = pd.DataFrame(np.array(DistilBERT_Output[0][:,0,:]))
features_df.shape
#features_df.index
#features_df.head(5)

# It's instructive to view the distribution of the hidden vector values for some of the data points
features_df.loc[0].describe()
plt.hist(features_df.loc[1])

# Create a label DataFrame
labels_df = df_Sentiment_Train[['Label']]
labels_df.shape
#labels_df.index
#labels_df.head(5)

#~~~~~~~~~~~~~~~~~~~~~~~~~
## Train classifier models
#~~~~~~~~~~~~~~~~~~~~~~~~~

# Create train/test splits
train_features_df, test_features_df, train_labels_df, test_labels_df = \
train_test_split(features_df, labels_df, train_size = 0.75, random_state=40)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using a Naive model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# View the distribution of labels in the test_label_df
test_labels_df.groupby(['Label']).size()

# Therefore a naive model that predicts everything as 0 will achieve 55% accuracy
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

log_reg_predictions.columns = ['Prediction', 'Prob_0', 'Prob_1']

# From the naive model - what's the probability of a random data point being 1
cut_off = test_labels_df[test_labels_df['Label']==1].shape[0]/test_labels_df.shape[0]

# Specify the tuned predictions based on this cut off
log_reg_predictions['Tuned_Prediction'] = (log_reg_predictions[['Prob_1']] >= cut_off).astype(int)

log_reg_predictions.describe()

from sklearn.metrics import accuracy_score

# Check the indicies are equal
test_labels_df.index.equals(log_reg_predictions.index)

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

# Use the Adam optimiser with the binary crossentropy (logloss) metric as the loss function
nnmodel.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

# View the network topology
print('Neural Network Model Summary: ')
print(nnmodel.summary())

# Convert labels with one-hot encoding
encoder = OneHotEncoder(sparse=False)

train_labels_cat = pd.DataFrame(encoder.fit_transform(train_labels_df).astype(int))
test_labels_cat = pd.DataFrame(encoder.fit_transform(test_labels_df).astype(int))
#type(train_labels_cat)
train_labels_cat.shape
test_labels_cat.shape

# Can pass DataFrames through the network
nnmodel_hist = nnmodel.fit(train_features_df, 
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

# Test the model on testing data
results = nnmodel.evaluate(test_features_df, test_labels_cat)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))


#~~~~~~~~~~~~~~~~~~~~~~
# Using a Random Forest
#~~~~~~~~~~~~~~~~~~~~~~
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence

import pandas_profiling as profiler
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline



from sklearn.ensemble import RandomForestClassifier

# Hyperparameters
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, train_size=0.70, random_state=11)

model = CatBoostClassifier(
    custom_loss=['Logloss'],
    eval_metric='Precision',
    random_seed=11,
    logging_level='Silent',
    nan_mode='Min')

model.fit(
    X_train, y_train,
    #cat_features=cat_features,
    eval_set=(X_cv, y_cv),
    #plot=True,
    use_best_model=True
)

print("On Training Data")
print(classification_report(y_train, model.predict(X_train)))
print("On Test Data")
print(classification_report(y_test, model.predict(X_test)))

# Random forest
# define random forest model
rand_forest_model = RandomForestClassifier(n_estimators=1000, max_depth=8,random_state=11,min_samples_leaf=20)
rand_forest_model.fit(X_train, y_train)

print("**On Training Data**")
print(classification_report(y_train, rand_forest_model.predict(X_train)))
print("**On Test Data**")
print(classification_report(y_test, rand_forest_model.predict(X_test)))

# Num of trees
n_estimators = [200,400,600,800,1000,1200,1400]
# Num of features at every split
max_features = ['auto', 'sqrt']
# Maximum tree depth
max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Out of bag scoring
oob_score = [True, False]

hyper_param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'oob_score': oob_score}

rf_random = RandomizedSearchCV(estimator = rand_forest_model, param_distributions = hyper_param_grid,
                               n_iter = 30, cv = 5, verbose=2, random_state=11, n_jobs = -1)

# Fit the random forest model on cross-validation dataset
rf_random.fit(X_cv, y_cv)

# best parameters from grid-search
rf_random.best_params_


rand_forest_model = RandomForestClassifier(n_estimators = rf_random.best_params_['n_estimators'],
                                           min_samples_split=rf_random.best_params_['min_samples_split'],
                                           min_samples_leaf = rf_random.best_params_['min_samples_leaf'],
                                           max_features = rf_random.best_params_['max_features'],
                                           max_depth = rf_random.best_params_['max_depth'],
                                           bootstrap = rf_random.best_params_['bootstrap'],
                                           oob_score = rf_random.best_params_['oob_score'],
                                           random_state=11,
                                           n_jobs=-1)
rand_forest_model.fit(X_train, y_train)

print("**On Training Data After CV Grid Search**")
print(classification_report(y_train, rand_forest_model.predict(X_train)))
print("**On Test Data After CV Grid Search**")
print(classification_report(y_test, rand_forest_model.predict(X_test)))



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using an extreme gradient boosted forest
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from xgboost import XGBClassifier


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Final comparison of classifier models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1. Naive
test_labels_df[test_labels_df['Label']==0].shape[0]/test_labels_df.shape[0]

# 2. Logistic regression
lr_clf.score(test_features_df, test_labels_df)
print(accuracy_score(test_labels_df, log_reg_predictions[['Tuned_Prediction']]))

# 3. Random forest

# 4. Xgboost

# 5. Neural network
nn_results = nnmodel.evaluate(test_features_df, test_labels_cat)
nn_results[1]
