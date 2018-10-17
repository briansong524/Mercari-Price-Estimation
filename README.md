# Mercari-Price-Estimation
Based on the Kaggle Competition "Mercari Price Suggestion Challenge"

## Summary

Using item description and meta-data, estimate the price of the listed items. Two methods were tried to compare and contrast: (1) ensemble method of Microsoft's [lightgbm](https://github.com/Microsoft/LightGBM) and scikit-learn Ridge Regression, and (2) Convolutional Neural Network (CNN) with Tensorflow. Used Kaggle's [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge) data to train and run the model. Used the test set RMSE to validate the efficacy of the models. 

## Dependencies

*Python 2.7* (should work for Python 3.x as well)

*pandas* - DataFrame for output 

*numpy* - To handle vectors

*glob* - Handle directories

*NLTK* - Tokenize text and general text preprocessing

*sklearn* - Ridge Regression, text handling, and general setup for model training

*scipy* - Handling large sparse data

*lightgbm* - Microsoft's lightweight gradient boost machine model

## Files

**Mercari Price Suggestion - Analysis .ipynb** - preliminary analysis of the data 

#### In directory 'ensemble_learning':

**cleaning.py** - cleans original training data and dumps important variables as multiple pickle files

**ensemble.py** - combines the two models' results with ensemble method and store the predictions in a pickle file

**ml_training.py** - train Ridge regression and lightgbm model

**utils.py** - many used methods stored in a separate file for cleanliness

#### In directory 'keras_cnn':

**main.py** - clean and train a CNN model.

**utils.py** - many used methods stored in a separate file for cleanliness

**keras cnn kaggle_person.ipynb** - reference code from some person's kernel

**keras cnn rnn,ipynb** - code used to play around with cnn and rnn

**tensorflow clean NN v05 - RNN.ipynb** - using Tensorflow to run RNN on text  

## How to use

First, get the training data from [here](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data) and extract/store it in the directory with the scripts. I probably could've done this much cleaner, but it was one of my earlier projects so I was not great at this.

For the ensemble learning directory, run cleaning.py -> ml_training.py -> ensemble.py.

For keras_cnn directory, just run main.py.

