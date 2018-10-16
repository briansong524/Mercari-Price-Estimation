#!/usr/bin/env python3

## clean "category_name" and make numeric indicies for one-worded features (brand_name, cat1/2/3)
__all__ = ['split_cat', 'handle_missing_inplace', 'just_punc_and_lower', 'build_dictionary', 'convert_word_to_padded', 'convert_word_to_ind',
			'']
from utils import *
import pickle
import pandas as pd
import numpy as np


## import data 
start_time = time.time()

train_raw = pd.read_csv('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/train.tsv',delimiter= '\t')
#train_raw = train_raw.iloc[0:10000,] # just a bit
# standardize price here because may as well
normalized_price = np.log1p(train_raw['price'].values)
mean_price_norm = np.mean(normalized_price)
std_price_norm = np.std(normalized_price) 
train_raw['price'] = (normalized_price - mean_price_norm)/std_price_norm 

end_time = time.time()
print('import data took ' + str(end_time - start_time) + " seconds.")

## clean dataframe

start_time = time.time()

train_raw['cat1'],train_raw['cat2'],train_raw['cat3'] = \
zip(*train_raw['category_name'].apply(lambda x: split_cat(x))) # split the categories into three new columns
train_raw.drop('category_name',axis = 1, inplace = True) # remove the column that isn't needed anymore

handle_missing_inplace(train_raw) # replaces NaN with a string placeholder 'missing'

end_time = time.time()
print('cleaning "category_name" and making one-worded features to indices took ' + str(end_time - start_time) + " seconds.")

## convert name and item_desc to indices, then configure a bit more

start_time = time.time()

train_raw['name_token'] = just_punc_and_lower(train_raw.name)
train_raw['itemdesc_token'] = just_punc_and_lower(train_raw.item_description)

end_time = time.time()
print('cleaning name and item_description and getting all the words took ' + str(end_time - start_time) + " seconds.")

print(' ')
print('pickling dataframe...')

train_raw.to_pickle('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/cleaned_train.pkl')

print(' ')
print('building dictionaries for "name" and "item_description"...')

## Recurrent NN setup for name and item_description

all_name = [item for sublist in train_raw.name_token for item in sublist]
all_item_desc = [item for sublist in train_raw.itemdesc_token for item in sublist]

dict_name_len = 12000
dict_itemdesc_len = 8737

name_pad_len = 7
itemdesc_pad_len = 20

name_embed_len = 15
itemdesc_embed_len = 15

# build dictionary for indices and use it to convert word to indices
# same time, the indices will be padded to maintain np.array dtype

name_dict, name_dict_rev = build_dictionary(all_name, dict_name_len)
name_padded, _ = convert_word_to_padded(train_raw.name_token, name_dict, name_pad_len)

itemdesc_dict, itemdesc_dict_rev = build_dictionary(all_item_desc, dict_itemdesc_len)
itemdesc_padded, _ = convert_word_to_padded(train_raw.itemdesc_token, itemdesc_dict, itemdesc_pad_len)

print(' ')
print('pickling name dictionary, reverse dictionary, and padded variable')

pickle.dump(name_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/name_dict.pkl','wb')
pickle.dump(name_dict_rev, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/name_dict_rev.pkl','wb')
pickle.dump(name_padded, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/name_padded.pkl','wb')

print('pickling item_description dictionary, reverse dictionary, and padded variable')

pickle.dump(itemdesc_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict.pkl','wb')
pickle.dump(itemdesc_dict_rev, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict_rev.pkl','wb')
pickle.dump(itemdesc_padded, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_padded.pkl','wb')

print(' ')
print('building dictionaries for brand_name and three categories...')
## make dictionaries for brand_name and cat1/2/3

start_time = time.time()
# define dictionary lengths here
dict_brand_len = 379 # total words + 1 for "UNK" is minimal size
dict_cat1_len = 12 
dict_cat2_len= 115 
dict_cat3_len = 305

brand_name_dict, brand_name_dict_rev = build_dictionary(train_raw['brand_name'], dict_brand_len)
train_raw['brand_name_inds'], count_unk_brand = convert_word_to_ind(train_raw['brand_name'].values.reshape((-1,1)), brand_name_dict)
cat1_dict ,cat1_rev_dict= build_dictionary(train_raw['cat1'],dict_cat1_len)
train_raw['cat1_inds'], count_unk_cat1 = convert_word_to_ind(train_raw['cat1'].values.reshape((-1,1)), cat1_dict)
cat2_dict ,cat2_rev_dict= build_dictionary(train_raw['cat2'],dict_cat2_len)
train_raw['cat2_inds'], count_unk_cat2 = convert_word_to_ind(train_raw['cat2'].values.reshape((-1,1)), cat2_dict)
cat3_dict ,cat3_rev_dict= build_dictionary(train_raw['cat3'],dict_cat3_len)
train_raw['cat3_inds'], count_unk_cat3 = convert_word_to_ind(train_raw['cat3'].values.reshape((-1,1)), cat3_dict)

print(str(count_unk_brand) + ' ' + str(count_unk_cat1) + ' '+ str(count_unk_cat2) + " " + str(count_unk_cat3))

end_time = time.time()
print('making dictionaries for brand and categories took ' + str(end_time - start_time) + " seconds.")

print(' ')
print('pickling brand_name dictionary and reverse dictionary')

pickle.dump(brand_name_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict.pkl','wb')
pickle.dump(brand_name_dict_rev, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict_rev.pkl','wb')

print('pickling cat1 dictionary and reverse dictionary')

pickle.dump(cat1_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict.pkl','wb')
pickle.dump(cat1_rev_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict_rev.pkl','wb')

print('pickling cat2 dictionary and reverse dictionary')

pickle.dump(cat2_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict.pkl','wb')
pickle.dump(cat2_rev_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict_rev.pkl','wb')

print('pickling cat3 dictionary and reverse dictionary')

pickle.dump(cat3_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict.pkl','wb')
pickle.dump(cat3_rev_dict, open('/home/bsong/Python_Stuff/Data/Kaggle_Mercari/itemdesc_dict_rev.pkl','wb')

