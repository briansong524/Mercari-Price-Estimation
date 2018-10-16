import numpy as np
import pandas as pd

import collections
import random
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import nltk
import itertools
import matplotlib.pyplot as plt
import re

## define functions to use

######## Basic text manipulation functions (some specific to Mercari Kaggle Competition) 

def split_cat(text): # this one is to reduce the categoriy_name into three subcategories
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

def handle_missing_inplace(dataset):  # this one is to put placeholders in place of missing values (NaN)
    dataset['cat1'].fillna(value='No Label', inplace=True)
    dataset['cat2'].fillna(value='No Label', inplace=True)
    dataset['cat3'].fillna(value='No Label', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='No description yet', inplace=True)
     
def build_dictionary(words, n_words): # dictionary that maps words to indices. this function should be modular.
    #input is [['a','b','c'],['a','b','c']]
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]] # word indexed as "unknown" if not one of the top #n_words (popular/common) words (-1 is filler #)
    count.extend(Counter(words).most_common(n_words - 1)) # most_common returns the top (n_words-1) ['word',count]
    dictionary = dict()
    for word, _ in count: # the 'word, _' is writted because count is a list of list(2), so defining 'word' as the first term per
        dictionary[word] = len(dictionary) # {'word': some number incrementing by one. fyi, no repeats because from most_common)}
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys())) # {ind. : 'word'} I guess for looking up if needed?
    return dictionary, reversed_dictionary

def clean_and_tokenize(dataset_col): # input is a column of strings
    pattern = '[A-Za-z]+' # does this only keep words
    pattern2 = '[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n]' # to rid of special characters
    list_of_lists = list()
    tokenizer = RegexpTokenizer(pattern)
    stop_words = set(stopwords.words('english'))
    for word in dataset_col:
        list_of_words = list()
        word = re.sub(pattern2, r'', word)
        tokenized = tokenizer.tokenize(word)
        tokenized_filtered = filter(lambda token: token not in stop_words, tokenized)
        for i in tokenized_filtered:
            if (len(i) > 2 ): #ignore words of length 2 or less
                list_of_words.append(i.lower()) # append all words to one list
        list_of_lists.append(list_of_words)
    list_as_series = pd.Series(list_of_lists)
    return list_as_series

def just_punc_and_lower(dataset_col):
    pattern = '[A-Za-z]+' # does this only keep words
    pattern2 = '[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n]' # to rid of special characters
    tokenizer = RegexpTokenizer(pattern)
    list_of_lists = list()
    for word in dataset_col:
        word = re.sub(pattern2,r'',word)
        list_of_words = word.lower().split(' ')
        list_of_lists.append(list_of_words)
    list_as_series = pd.Series(list_of_lists)
    return list_as_series

def convert_word_to_ind(dataset_col,dictionary): # input the pandas column of texts and dictionary. This should be modular
    # each input should be a string of cleaned words tokenized into a list (ex. ['this', 'is', 'an', 'item'])
    # dictionary should be the dictionary obtained from build_dictionary
    list_of_lists = []
    unk_count = 0 # total 'unknown' words counted
    for word_or_words in dataset_col: # words is the list of all words
        list_of_inds = []
        for word in word_or_words:
            if word in dictionary:
                index = np.int(dictionary[word]) # dictionary contains top words, so if in, it gets an index
            else:
                index = 0  #  or dictionary['UNK']? can figure out later
                unk_count += 1
            list_of_inds.append(index)
        list_of_lists.append(list_of_inds)

    # make list_of_lists into something that can be put into pd.DataFrame
    #list_as_series = pd.Series(list_of_lists)
    list_as_series = np.array(list_of_lists)
    return list_as_series, unk_count

def pad_word_indices(col_of_indices, pad_length): # col_of_indices can be a pd series. 
    temp_series = [] # append vectors into here
    for list_inds in col_of_indices:
        len_list = len(list_inds)
        if len_list >= pad_length:
            temp_series.append(np.array(list_inds[(len_list-pad_length):]))
        else:
            padded_vec = [0]*(pad_length-len_list)
            padded_vec.extend(list_inds)
            temp_series.append(np.array(padded_vec))
    return temp_series

def convert_word_to_padded(dataset_col,dictionary,pad_length): # input the pandas column of texts and dictionary. This should be modular
    # each input should be a string of cleaned words tokenized into a list (ex. ['this', 'is', 'an', 'item'])
    # dictionary should be the dictionary obtained from build_dictionary
    # use this function when you know how long you want your pad_length
    #   - otherwise, use convert_word_to_ind, and pad_word_indices
    #   - eventually, will look into cleaning these three functions up.
    list_of_lists = []
    unk_count = 0 # total 'unknown' words counted
    for word_or_words in dataset_col: # words is the list of all words
        list_of_inds = []
        count_inds = 0
        for word in word_or_words:
            if word in dictionary:
                index = np.int(dictionary[word]) # dictionary contains top words, so if in, it gets an index
            else:
                index = 0  #  or dictionary['UNK']? can figure out later
                unk_count += 1
            count_inds +=1
            list_of_inds.append(index) 
        if count_inds >= pad_length:
            asdf = list_of_inds[(count_inds-pad_length):]
        else: 
            asdf = [0]*(pad_length-count_inds)
            asdf.extend(list_of_inds)
        temp = np.array(asdf)
        list_of_lists.append(temp)
    list_as_series = np.array(list_of_lists)
    return list_as_series, unk_count

def obtain_reasonable_vocab_size(list_words, perc_words = .95):
    counter_ = Counter(list_words)
    counts = [i for _,i in counter_.most_common()]
    tot_words = len(list_words)
    print('total words (with repeats): ' + str(tot_words))
    tot_count = 0
    runs = 0
    while tot_count < round(perc_words*tot_words):
        tot_count += counts[runs]
        runs += 1
    print('reasonable vocab size: ' + str(runs))

def obtain_reasonable_pad_length(list_words, perc_words = 0.95):
    len_list = [len(i) for i in list_words]
    sort_list = sorted(len_list)
    ind_good = sort_list[round(perc_words*len(sort_list))]
    print('reasonable pad length: ' + str(ind_good))
    

###################################################################################################################
