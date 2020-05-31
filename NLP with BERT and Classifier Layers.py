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

#########################
# STEP 1: Load BERT model
#########################

#~~~~~~~~~~~~~~~~~~~~~~~~~
## Bring in data and clean
#~~~~~~~~~~~~~~~~~~~~~~~~~
# Originally located below I have included a copy in my repo
# https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/blob/master/data/stsa.binary.train

df_Sentiment_Train_Full = \
pd.read_csv("https://github.com/ThomasHandscomb/NLP-with-BERT/raw/master/train.csv"                                 
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
# Define which NLP model and pre-trained weights to use. In this example I use the light-weight version of
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

print(type(DistilBERT_Output[0]))
print(padded_tokenized_text_tensor.shape)
print(DistilBERT_Output[0].shape)

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

####################################
# STEP 2: Final Classification Layer
####################################

# Create train/test splits
train_features_df, test_features_df, train_labels_df, test_labels_df = \
train_test_split(features_df, labels_df, train_size = 0.75, random_state=40)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Using a Naive model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# View the distribution of labels in the test_label_df
test_labels_df.groupby(['Label']).size()

# Therefore a naive model that predicts everything as 0 will achieve 55% accuracy
test_labels_df[test_labels_df['Label']==0].shape[0]/test_labels_df.shape[0]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Using a logistic regression
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lr_clf = LogisticRegression()
lr_clf.fit(train_features_df, train_labels_df)

from sklearn.metrics import classification_report

print("Classification On Test Data at default probability threshold (50%)")
print(classification_report(test_labels_df, lr_clf.predict(test_features_df)))

#~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Using a Random Forest
#~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define random forest classifier model
rand_forest_model = RandomForestClassifier(n_estimators=1000
                                           , max_depth=8
                                           , random_state=11
                                           , min_samples_leaf=20)

# Specify hyperparameters to be used in a grid search

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
               'bootstrap': bootstrap}
               #'oob_score': oob_score}

# Specify a brute force search over hyperparameter grid
rf_random = RandomizedSearchCV(estimator = rand_forest_model
                               , param_distributions = hyper_param_grid
                               , n_iter = 30
                               , cv = 5
                               , verbose=2
                               , random_state=11
                               , n_jobs = -1)

# Create hyperparameter training set from the training set
hyper_train_features_df, hyper_tuning_features_df, hyper_train_labels_df, hyper_tuning_labels_df = \
train_test_split(train_features_df, train_labels_df, train_size = 0.75, random_state=40)

# Fit the random forest model on hyperparameter tuning dataset
rf_random.fit(hyper_tuning_features_df, hyper_tuning_labels_df)

# Best parameters from grid-search
rf_random.best_params_

# Define forest model using the tuned hyperparameters
random_forest_model = RandomForestClassifier(n_estimators = rf_random.best_params_['n_estimators'],
                                           min_samples_split=rf_random.best_params_['min_samples_split'],
                                           min_samples_leaf = rf_random.best_params_['min_samples_leaf'],
                                           max_features = rf_random.best_params_['max_features'],
                                           max_depth = rf_random.best_params_['max_depth'],
                                           bootstrap = rf_random.best_params_['bootstrap'],
                                           #oob_score = rf_random.best_params_['oob_score'],
                                           random_state=11,
                                           n_jobs=-1)
# Fit tuned model to the training set
random_forest_model.fit(train_features_df, train_labels_df)

# Compare to a non-tuned version
basic_random_forest_model = RandomForestClassifier()
basic_random_forest_model.fit(train_features_df, train_labels_df)

# Examine simple classifications at default threshold (50%)
print("Tuned Random Forest On Test Data After CV Grid Search")
print(classification_report(test_labels_df, random_forest_model.predict(test_features_df)))

print("Basic Random Forest On Test Data After CV Grid Search")
print(classification_report(test_labels_df, basic_random_forest_model.predict(test_features_df)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Using an extreme gradient boosted model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note: Gradient boosting builds 1 weak learner (typically a tree) at a time and improves this 
# by gradient decent. A random forest builds each weak learner independently and then aggregates at the 
# end (by majority (classification) or averaging (regression)) to form the strong learner

from xgboost import XGBClassifier

# Recall cv/hyper parameter tuning data sets
#hyper_train_features_df, hyper_tuning_features_df, hyper_train_labels_df, hyper_tuning_labels_df

xgb_model = XGBClassifier(
    eval_metric='logloss',
    random_seed=11,
    logging_level='Silent',
    nan_mode='Min')

evaluation_set = [(hyper_train_features_df, hyper_train_labels_df), (hyper_tuning_features_df, hyper_tuning_labels_df)]

xgb_model.fit(hyper_train_features_df
              , hyper_train_labels_df
              , eval_set=evaluation_set
              )

print("XGboost On Test Data")
print(classification_report(test_labels_df, xgb_model.predict(test_features_df)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. Using a Neural Network
#~~~~~~~~~~~~~~~~~~~~~~~~~~

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

# Pass DataFrames through the network
nnmodel_hist = nnmodel.fit(train_features_df, 
                           train_labels_cat, 
                           shuffle=True,
                           #validation_split=0.3, 
                           verbose=2, 
                           batch_size=10, 
                           epochs=30)

#print(nnmodel_hist.history.keys())

# View the output plotted at each epoch

# Summarize history for accuracy
plt.figure(figsize=(15,7.5))
plt.plot(nnmodel_hist.history['accuracy'])
#plt.plot(nnmodel_hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Summarize history for loss
plt.figure(figsize=(15,7.5))
plt.plot(nnmodel_hist.history['loss'])
#plt.plot(nnmodel_hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Test the network model on testing data
print("Neural Net Performance on Test Data")
print(classification_report(test_labels_df, nnmodel.predict_classes(test_features_df)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Final comparison of classifier models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#~~~~~~~~~~~~~~~~~~
# Simple accuracies
#~~~~~~~~~~~~~~~~~~

# 1. Naive
test_labels_df[test_labels_df['Label']==0].shape[0]/test_labels_df.shape[0]

# 2. Logistic regression
print("Logistic Regression at 50% cutoff")
print(classification_report(test_labels_df, lr_clf.predict(test_features_df)))

# 3. Random forest
print("Tuned Random Forest Model")
print(classification_report(test_labels_df, random_forest_model.predict(test_features_df)))

print("Basic Random Forest On Test Data After CV Grid Search")
print(classification_report(test_labels_df, basic_random_forest_model.predict(test_features_df)))

# 4. Gradient Boosted (Xgboost)
print("Xgboost model")
print(classification_report(test_labels_df, xgb_model.predict(test_features_df)))

# 5. Neural network
nn_results = nnmodel.evaluate(test_features_df, test_labels_cat)
nn_results[1]

print("Neural Network")
print(classification_report(test_labels_df, nnmodel.predict_classes(test_features_df)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Precision/Recall comparisons
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.metrics import precision_recall_curve
#~~~~~~~~~~~~~~~~~~~~~
# Logistic Regression:
log_reg_preds_probs = pd.DataFrame(lr_clf.predict_proba(test_features_df), index=test_features_df.index)
# keep probabilities for the positive outcome only
log_reg_preds_probs_1s = log_reg_preds_probs.iloc[:,1]
lr_precision, lr_recall, lr_thresholds = precision_recall_curve(test_labels_df, log_reg_preds_probs_1s)
# AUC
lr_auc = roc_auc_score(test_labels_df, log_reg_preds_probs_1s)

# Calculate roc curves
lr_fpr, lr_tpr, lr_thresholds = roc_curve(test_labels_df, log_reg_preds_probs_1s)

#~~~~~~~~~~~~~~~~~~~~~
# Random Forest
random_forest_preds_probs = pd.DataFrame(random_forest_model.predict_proba(test_features_df), index=test_features_df.index)
# keep probabilities for the positive outcome only
random_forest_preds_probs_1s = random_forest_preds_probs.iloc[:,1]
rf_precision, rf_recall, rf_thresholds = precision_recall_curve(test_labels_df, random_forest_preds_probs_1s)
#AUC
rf_auc = roc_auc_score(test_labels_df, random_forest_preds_probs_1s)

# calculate roc curves
rf_fpr, rf_tpr, rf_thresholds = roc_curve(test_labels_df, random_forest_preds_probs_1s)

# Can compare this to the non-tuned random forest model
basic_random_forest_preds_probs = pd.DataFrame(basic_random_forest_model.predict_proba(test_features_df), index=test_features_df.index)
# keep probabilities for the positive outcome only
basic_random_forest_preds_probs_1s = basic_random_forest_preds_probs.iloc[:,1]
basic_rf_precision, basic_rf_recall, basic_rf_thresholds = precision_recall_curve(test_labels_df, basic_random_forest_preds_probs_1s)
#AUC
basic_rf_auc = roc_auc_score(test_labels_df, basic_random_forest_preds_probs_1s)

# calculate roc curves
basic_rf_fpr, basic_rf_tpr, basic_rf_thresholds = roc_curve(test_labels_df, basic_random_forest_preds_probs_1s)



#~~~~~~~~~~~~~~~~~~~~~
# Xgboost
xgb_preds_probs = pd.DataFrame(xgb_model.predict_proba(test_features_df), index=test_features_df.index)
# keep probabilities for the positive outcome only
xgb_preds_probs_1s = xgb_preds_probs.iloc[:,1]
xgb_precision, xgb_recall, xgb_thresholds = precision_recall_curve(test_labels_df, xgb_preds_probs_1s)
#AUC
xgb_auc = roc_auc_score(test_labels_df, xgb_preds_probs_1s)
# calculate roc curves
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(test_labels_df, xgb_preds_probs_1s)

#~~~~~~~~~~~~~~~~~~~~~
# Neural Net
nn_preds_probs = pd.DataFrame(nnmodel.predict_proba(test_features_df), index=test_features_df.index)
# keep probabilities for the positive outcome only
nn_preds_probs_1s = nn_preds_probs.iloc[:,1]
nn_precision, nn_recall, nn_thresholds = precision_recall_curve(test_labels_df, nn_preds_probs_1s)
#AUC
nn_auc = roc_auc_score(test_labels_df, nn_preds_probs_1s)
# calculate roc curves
nn_fpr, nn_tpr, nn_thresholds = roc_curve(test_labels_df, nn_preds_probs_1s)

#~~~~~~
# Plots
#~~~~~~
plt.style.use('ggplot')

#~~~~~~~~~~~~~~~~~~~~
# Plot the ROC curves
plt.figure(figsize=(15,7.5))
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
plt.plot(rf_fpr, rf_tpr, marker='.', label='Tuned Random Forest')
plt.plot(basic_rf_fpr, basic_rf_tpr, marker='.', label='Basic Random Forest')
plt.plot(xgb_fpr, xgb_tpr, marker='.', label='Xgboost')
plt.plot(nn_fpr, nn_tpr, marker='.', label='Neural Network')
# plot labels
plt.title('ROC curves for different classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

#~~~~~~~
# AUC
print('Logistic Regression', lr_auc)
print('Tuned Random Forest', rf_auc)
print('Basic Random Forest', basic_rf_auc)
print('Xgboost', xgb_auc)
print('Neural Net', nn_auc)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the precision-recall curves
plt.figure(figsize=(15,7.5))
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic Regression')
plt.plot(rf_recall, rf_precision, marker='.', label='Tuned Random Forest')
plt.plot(basic_rf_recall, basic_rf_precision, marker='.', label='Tuned Random Forest')
plt.plot(xgb_recall, xgb_precision, marker='.', label='Xgboost')
plt.plot(nn_recall, nn_precision, marker='.', label='Neural Network')
# plot labels
plt.title('Precision vs. Recall for different classifiers')
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()
