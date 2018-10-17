from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import numpy as np
import tensorflow as tf
import utils

FLAGS = None

def main():
	## import data 
	train_raw = pd.read_csv('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/train.tsv',delimiter= '\t')
	normalized_price = np.log1p(train_raw['price'].values)
	mean_price_norm = np.mean(normalized_price)
	std_price_norm = np.std(normalized_price) 
	train_raw['price'] = (normalized_price - mean_price_norm)/std_price_norm 

	# split the categories into three new columns
	train_raw['cat1'],train_raw['cat2'],train_raw['cat3'] = zip(*train_raw['category_name'].apply(lambda x: utils.split_cat(x))) 

	# remove the column that isn't needed anymore
	train_raw.drop('category_name',axis = 1, inplace = True) 

	# replaces NaN with a string placeholder 'missing'
	# note: this is mildly hardcoded so it has to come after splitting categories into three
	handle_missing_inplace(train_raw) 

	# make a dictionary for both name and item_description (figured similar words appear, so combining words from both)

	all_name_desc = np.hstack((train_raw['name'],train_raw['item_description'])) # get all dem words
	all_name_desc = utils.clean_and_tokenize(all_name_desc)
	all_name_desc = [item for sublist in all_name_desc for item in sublist]
	train_raw['name'] = utils.clean_and_tokenize(train_raw['name'])
	train_raw['item_description'] = utils.clean_and_tokenize(train_raw['item_description'])

	# Build dictionaries here
	vocabulary_size = 100000 # keeping 100000 words in the dictionary. 0.28% of total words were put into "UNK". so kept 99.72% "common" words
	word2vec_dict, reverse_dict = utils.build_dictionary(all_name_desc,vocabulary_size) 

	dict_brand_len = 3000 # .16% of the words were put into "UNK"
	dict_cat1_len = 12 # theres apparently less than 12 categories in cat1
	dict_cat2_len= 100 # .114% of the words were put into "UNK"
	dict_cat3_len = 700 # .04% of works were put into "UNK"

	brand_name_dict, brand_name_dict_rev = utils.build_dictionary(train_raw['brand_name'], dict_brand_len)
	train_raw['brand_name_inds'], count_unk_brand = utils.convert_word_to_ind(train_raw['brand_name'].values.reshape((-1,1)), brand_name_dict)
	cat1_dict ,cat1_rev_dict= utils.build_dictionary(train_raw['cat1'],dict_cat1_len)
	train_raw['cat1_inds'], count_unk_cat1 = utils.convert_word_to_ind(train_raw['cat1'].values.reshape((-1,1)), cat1_dict)
	cat2_dict ,cat2_rev_dict= utils.build_dictionary(train_raw['cat2'],dict_cat2_len)
	train_raw['cat2_inds'], count_unk_cat2 = utils.convert_word_to_ind(train_raw['cat2'].values.reshape((-1,1)), cat2_dict)
	cat3_dict ,cat3_rev_dict= utils.build_dictionary(train_raw['cat3'],dict_cat3_len)
	train_raw['cat3_inds'], count_unk_cat3 = utils.convert_word_to_ind(train_raw['cat3'].values.reshape((-1,1)), cat3_dict)

	# make some padded vectors and NOT store them back in pandas df (keeping it as np.array)
	name_pad_size = 9 # max length of name
	itemdesc_pad_size = 75 # 95th percentile of length of item descriptions

	name_padded , _ = utils.convert_word_to_padded(train_raw.name,word2vec_dict,name_pad_size) # without _, will get tuple lol.
	itemdesc_padded , _ = utils.convert_word_to_padded(train_raw.item_description,word2vec_dict,itemdesc_pad_size) 

	# Define some embedding lengths here
	name_emb_size = 15
	itemdesc_emb_size = 15
	brand_emb_size = 10
	cat1_emb_size = 10
	cat2_emb_size = 10
	cat3_emb_size = 10
	itemcond_emb_size = 10
	shipping_emb_size = 10

	# lengths needed here and a bit later
	itemcond_len = np.max(train_raw.item_condition_id.values)

	name_itemdesc_emb = embed([i for i in range(vocabulary_size)],vocabulary_size,name_emb_size, name= 'name_itemdesc_emb')
	brand_emb = embed(train_raw.brand_name_inds,dict_brand_len, brand_emb_size, name= 'brand_emb')
	cat1_emb = embed(train_raw.cat1_inds,dict_cat1_len,cat1_emb_size, name= 'cat1_emb')
	cat2_emb = embed(train_raw.cat2_inds,dict_cat2_len,cat2_emb_size, name= 'cat2_emb')
	cat3_emb = embed(train_raw.cat3_inds,dict_cat3_len,cat3_emb_size, name= 'cat3_emb')
	itemcond_emb = embed(train_raw.item_condition_id,itemcond_len ,itemcond_emb_size, name= 'itemcond_emb')
	shipping_emb = embed(train_raw.shipping, 2, shipping_emb_size, name= 'shipping_emb')

	# Setup feeding stuff here

	# somewhat state which variables will be used here
	# reshaped to fit better (not sure if too necessary in hindsight, but minimal loss in time)
	input_name = name_padded
	input_itemdesc = itemdesc_padded
	input_price = train_raw['price'].values.reshape((-1,1))
	input_brand = train_raw.brand_name_inds.values.reshape((-1,1))
	input_cat1 = train_raw.cat1_inds.values.reshape((-1,1))
	input_cat2 = train_raw.cat2_inds.values.reshape((-1,1))
	input_cat3 = train_raw.cat3_inds.values.reshape((-1,1))
	input_itemcond = train_raw.item_condition_id.values.reshape((-1,1))
	input_ship = train_raw.shipping.values.reshape((-1,1))

	# define some lengths for partitioning data after feeding
	input_name_len = input_name.shape[1]
	input_itemdesc_len = input_itemdesc.shape[1]

	# concatenate data to make into tensor slices
	temp_set = np.concatenate((input_name, input_itemdesc,input_cat1,input_cat2,input_cat3,
	                           input_brand, input_itemcond, input_ship),axis = 1) #name_and_desc ,input_itemcond,input_shipping
	shape_set = temp_set.shape[1] 
	batch_len = 10000

	num_epoch = 25
	tot_iter = train_raw.shape[0]* num_epoch // batch_len + 1

	print('splitting labels and features...')
	features_input = temp_set.astype(np.int32)
	label_input = input_price.astype(np.float32)
	# make some placeholders to avoid GraphDef exceeding 2GB
	feat_placeholder = tf.placeholder(features_input.dtype, features_input.shape)
	label_placeholder = tf.placeholder(label_input.dtype, label_input.shape)
	print('making tensor slices...')
	dataset = tf.data.Dataset.from_tensor_slices((feat_placeholder, label_placeholder))
	print('shuffling...')
	#np.random.shuffle(temp_set) # shuffle the data
	dataset = dataset.shuffle(buffer_size =10000)
	print('making epochs...')
	dataset = dataset.repeat(num_epoch) # epoch
	print('making batches...')
	dataset = dataset.batch(batch_len) 
	iterator = dataset.make_initializable_iterator()
	next_batch = iterator.get_next()

	# Tensorflow model setup

	input_x = tf.placeholder(tf.int32,[None, shape_set], name = "input_x") # pad_length = 25 or something defined earlier
	input_y = tf.placeholder(tf.float32,[None,1], name = "input_y") # train agianst this


	input_x_name = input_x[:,:input_name_len]
	input_x_itemdesc = input_x[:,input_name_len:(input_name_len + input_itemdesc_len)]
	input_x_cat1 = input_x[:,(input_name_len + input_itemdesc_len)]
	input_x_cat2 = input_x[:,(input_name_len + input_itemdesc_len)+1]
	input_x_cat3 = input_x[:,(input_name_len + input_itemdesc_len)+2]
	input_x_brand = input_x[:,(input_name_len + input_itemdesc_len)+3]
	input_x_itemcond = input_x[:,(input_name_len + input_itemdesc_len)+4]
	input_x_shipping = input_x[:,(input_name_len + input_itemdesc_len)+5]


	name_emb_lookup = tf.nn.embedding_lookup(name_itemdesc_emb, input_x_name)
	itemdesc_emb_lookup = tf.nn.embedding_lookup(name_itemdesc_emb,input_x_itemdesc)
	brand_emb_lookup = tf.nn.embedding_lookup(brand_emb,input_x_brand)
	cat1_emb_lookup = tf.nn.embedding_lookup(cat1_emb,input_x_cat1)
	cat2_emb_lookup = tf.nn.embedding_lookup(cat2_emb,input_x_cat2)
	cat3_emb_lookup = tf.nn.embedding_lookup(cat3_emb,input_x_cat3)
	itemcond_emb_lookup = tf.nn.embedding_lookup(itemcond_emb, input_x_itemcond)
	shipping_emb_lookup = tf.nn.embedding_lookup(shipping_emb, input_x_shipping)

	# expand name and item_desc because conv2d wants it 4-d
	name_emb_lookup_expand = tf.expand_dims(name_emb_lookup,-1)
	itemdesc_emb_lookup_expand = tf.expand_dims(itemdesc_emb_lookup,-1)

	# set some lazy parameters here
	out_nodes = 15
	dropout_keep_prob = tf.placeholder(tf.float32)

	W_shape_name = [1,name_emb_size,1,out_nodes] #figure this out if it works
	b_shape_name = out_nodes # same as last dimension in W

	W_shape_itemdesc = [1,itemdesc_emb_size,1,out_nodes]
	b_shape_itemdesc = out_nodes

	#layers_namedesc = test_cnn(input_x_namedesc,W_shape_namedesc,b_shape_namedesc,dropout_keep_prob)
	layers_name = CNN(name_emb_lookup_expand,W_shape_name,b_shape_name,dropout_keep_prob,name_pad_size)
	layers_itemdesc = CNN(itemdesc_emb_lookup_expand,W_shape_itemdesc,b_shape_itemdesc,dropout_keep_prob,itemdesc_pad_size)
	layers_brand = RegNN(brand_emb_lookup, dropout_keep_prob, dict_brand_len, brand_emb_size, batch_len, out_nodes)
	layers_cat1 = RegNN(cat1_emb_lookup, dropout_keep_prob, dict_cat1_len, cat1_emb_size, batch_len, out_nodes)
	layers_cat2 = RegNN(cat2_emb_lookup, dropout_keep_prob, dict_cat2_len, cat2_emb_size, batch_len, out_nodes)
	layers_cat3 = RegNN(cat3_emb_lookup, dropout_keep_prob, dict_cat3_len, cat3_emb_size, batch_len, out_nodes)
	layers_itemcond = RegNN(itemcond_emb_lookup, dropout_keep_prob, itemcond_len, itemcond_emb_size, batch_len, out_nodes)
	layers_shipping = RegNN(shipping_emb_lookup, dropout_keep_prob, 2, shipping_emb_size, batch_len, out_nodes)
	comb_layers = tf.concat([layers_name,layers_itemdesc, layers_brand, layers_cat1, 
	                         layers_cat2, layers_cat3,layers_itemcond, layers_shipping],axis=1) #, input_x_name, input_x_shipping

	#dense 
	dense1 = dense_NN(comb_layers, 64, batch_len)
	dense2 = dense_NN(dense1, 128, batch_len)
	predictions = dense_NN(dense2, 1, batch_len) 

	loss = 2
	loss,train_step  = train_the_NN(predictions,input_y,loss)
	# as is, normalized predictions cause NaN in rmsle solving. adding .00001 just in case
	unwind_true = tf.log(tf.expm1((input_y* std_price_norm) + mean_price_norm)+ .00001) 
	unwind_pred = tf.log(tf.expm1((predictions* std_price_norm) + mean_price_norm)+ .00001) 
	rmsle_ = tf.sqrt(tf.reduce_mean(tf.square(unwind_true - unwind_pred)))

	# Training model starts here
	with tf.Session() as sess:
    sess.run(iterator.initializer, {feat_placeholder: features_input, label_placeholder: label_input})
    init = tf.global_variables_initializer()
    sess.run(init)  
    i = 1
    rmsle_all = []
    for i in range(1,tot_iter):
        features_, label_ = sess.run(next_batch)

        sess.run(train_step,{input_x: features_, input_y: label_, dropout_keep_prob:.7})

        rmsle_solve = sess.run(rmsle_,{input_x: features_, input_y: label_,dropout_keep_prob:1})
        rmsle_all.append(rmsle_solve)
        end_time = time.time()
        if i % 50 == 0:
            print('running step: ' + str(i))
            loss_ = sess.run(loss,{input_x: features_, input_y: label_, dropout_keep_prob:1})
            print("average rmsle (of last 50 batches): %5.3f " % np.mean(rmsle_all))
            print("loss: " + str(loss_))
            tot_time = end_time - start_time
            print('fifty steps took %5.3f seconds.' % tot_time)
            print(' ')
            start_time = time.time()
            rmsle_all = []
        i += 1

    print('Done!')