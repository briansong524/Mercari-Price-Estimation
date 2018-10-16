import numpy as np
import pandas as pd
import tensorflow as tf
import collections
import random
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

######## Basic text manipulation functions (some specific to Mercari Kaggle Competition) 

def split_cat(text): 
	# this one is to reduce the categoriy_name into three subcategories
	# the text input looks like "Women/pants/blue" or something like that
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

def handle_missing_inplace(dataset):  
	# this one is to put placeholders in place of missing values (NaN).
    dataset['cat1'].fillna(value='No Label', inplace=True)
    dataset['cat2'].fillna(value='No Label', inplace=True)
    dataset['cat3'].fillna(value='No Label', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='No description yet', inplace=True)
     
def build_dictionary(words, n_words): 
	# dictionary that maps words to indices. this function should be modular.
    #input is [['a','b','c'],['a','b','c']]
    count = [['UNK', -1]] # word indexed as "unknown" if not one of the top #n_words (popular/common) words
    count.extend(Counter(words).most_common(n_words - 1)) # most_common returns the top (n_words-1) ['word',count]
    dictionary = dict()
    for word, _ in count: # the 'word, _' is writted because count is a list of list(2), so defining 'word' as the first term per
        dictionary[word] = len(dictionary) # {'word': some number incrementing by one. fyi, no repeats because from most_common)}
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys())) # {ind. : 'word'} I guess for looking up if needed?
    return dictionary, reversed_dictionary

def clean_and_tokenize(dataset_col): 
	# input is a column of strings
    pattern = '[A-Za-z]+' # does this only keep words? Anyways, can optimize so much here unique to different datasets
    list_of_lists = list()
    tokenizer = RegexpTokenizer(pattern)
    for word in dataset_col:
        list_of_words = list()
        tokenized = tokenizer.tokenize(word)
        for i in tokenized:
            if (len(i) > 2 ): #ignore words of length 2 or less
                list_of_words.append(i.lower()) # append all words to one list
        list_of_lists.append(list_of_words)
    list_as_series = pd.Series(list_of_lists)
    return list_as_series

def convert_word_to_ind(dataset_col,dictionary): 
    '''
    Input the pandas column of texts and dictionary. This should be modular
    Each input should be a string of cleaned words tokenized into a list (ex. ['this', 'is', 'an', 'item'])
    Dictionary should be the dictionary obtained from build_dictionary
    This and "pad_word_indices" should only be used during analysis (like finding optimal pad size and otherwise)
    When combining lists of different lengths (as is the output of this), it's a pain to convert it into a np.array,
    	even after padding to be equally lengthed. 
    After finding out the optimal padding length, run "convert_word_to_padded", which preemptively pads before appending 
    	the list of indices to a bigger list. This is easy to convert to np.array
    Personally found optimal padding lengths by doing something like:
		list_of_inds = convert_word_to_ind(blah, blah_dict)
    	a = [len(i) for i in list_of_inds]
    	print(np.max(a))
		plt.hist(a,bins = 20)
		perc_data = 0.95
		sorted_a = sorted(a)
		val = sorted_a[round(perc_data*len(sorted_a))]
		print(val) # val represents (perc_data)% of the length of a are under (val) words 
	Still useful in itself for one worded columns (brand_name, cat1/2/3), as it just functions similarly to LabelEncoder()
    '''
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

def pad_word_indices(col_of_indices, pad_length): 
	'''
	col_of_indices can be a pd series. 
    In hindsight, maybe this function isnt necessary.
    	After using "convert_word_to_ind", can just use the lengths found there and jump straight into "convert_word_to_padded"
    '''
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

def convert_word_to_padded(dataset_col,dictionary,pad_length): 
	'''
	input the pandas column of texts and associated dictionary. This should be modular
    each input should be a string of cleaned words tokenized into a list (ex. ['this', 'is', 'an', 'item'])
    dictionary should be the dictionary obtained from build_dictionary
    use this function when you know how long you want your pad_length
      - otherwise, use 'convert_word_to_ind', and find via how I explained there or otherwise.
      - eventually, will look into cleaning these three functions up.
    '''
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

######## Basic text manipulation functions (some specific to Mercari Kaggle Competition) 

def RegNN(x, dropout_keep_prob, vocab_size, embed_size, batch_len, out_len):
	'''
	Regular neural network function define here
	This is to use for the simpler columns (brand_name, item_condition, cat1/2/3)
	Note to self: maybe separating dropout is better for manipulation purposes (and pooling and dropout lol.)

	RegNN used for converting embedded features into whatever out_nodes. 
	I feel 'dense_NN' achieves the exact same thing, but one layer, but this happened because I was iteratively progressing through this 
	project and didn't want to erase too many things. 1/11/18
	
	x should be of size [batch_len,embed_size] 

	Update 1/12/18:
		I just realized why this function was so screwy. Prob might drop this
		out_len is just vocab_size. 'out_len' not even used here lol. 
	'''

   
    # set up some weights/bias stuff
    W1 = tf.Variable(tf.truncated_normal([vocab_size,embed_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[vocab_size,1]))  
    
    # Wx + b 
    NN_layer = tf.matmul(W1,tf.transpose(x)) + b1 # this outputs shape (vocab_size,batch_len)
    #print('NN_layer shape: ' + str(NN_layer.shape)) 
    # ReLU layer
    
    h = tf.nn.relu(NN_layer)
    
    # Drop Layer
    h_drop = tf.nn.dropout(h, dropout_keep_prob) # still (vocab_size,batch_len)
    
    return h_drop 


def embed(inputs, size, dim,name):
    # inputs is a list of indices
    # size is the number of unique indices (look for max index to achieve this if ordered)
    # dim is the number of embedded numbers 
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
    lookup = tf.nn.embedding_lookup(emb, inputs,name = name)
    #print(lookup.shape)
    return lookup

def CNN(x,W_shape,b_shape,dropout_keep_prob,pad_length):
	'''
	based on http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
	note to self: maybe separating dropout is better for manipulation purposes (and pooling and dropout lol.)
	x is the expanded lookup tables that will be trained
	'''
    
    W1 = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name="W1")
    b1 = tf.Variable(tf.constant(0.1, shape=[b_shape]), name="b1")
    conv = tf.nn.conv2d( #tf.layers.conv2d is also used, with more parameters. Probably a slightly higher API because of that.
        x,
        W1,
        strides = [1,1,1,1],
        padding="VALID",
        name="conv")
    
    h = tf.nn.relu(tf.nn.bias_add(conv, b1), name="relu")

    # pooling layer 
    pooled = tf.nn.max_pool(
                h,
                ksize=[1, pad_length, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

    pool_flat = tf.reshape(pooled, [-1, out_nodes])

    # Add dropout

    h_drop = tf.nn.dropout(pool_flat, dropout_keep_prob)

    return h_drop
    
    
def dense_NN(x,out_len,batch_len):
	'''
	As mentioned earller, this and RegNN might be redundant. If anything, this one is more clean and more correct.
		Not to mention, smaller 
	'''
    tot_nodes = x.shape[1]
    W_dense = tf.Variable(tf.truncated_normal([int(tot_nodes) , out_len], stddev=0.1), name="W2")
    b_dense = tf.Variable(tf.constant(0.1, shape=[batch_len,1]), name="b2")

    dense_out = tf.matmul(x,W_dense) + b_dense

    return dense_out

def train_the_NN(outnode,true_val,loss_val):
    loss_ = tf.sqrt(tf.losses.mean_squared_error(true_val, outnode))
    if loss_val > .7: # lol idk. 
        train_step_ = tf.train.AdamOptimizer(learning_rate = .001).minimize(loss_)
    else:
        train_step_ = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(loss_)
    return loss_, train_step_
