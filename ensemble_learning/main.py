
import time
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

from scipy.sparse import csr_matrix, hstack

from utils import *

'''
adsfpoiu is the input data pseudonym (so i can figure out how to deal with this later). was originally train_raw

'''

def main():
	start_time = time.time()

	handle_missing_inplace(asdfpoiu)
	print('[{}] Finished to handle missing'.format(time.time() - start_time))

	cutting(asdfpoiu)
	print('[{}] Finished to cut'.format(time.time() - start_time))

	to_categorical(asdfpoiu)
	print('[{}] Finished to convert categorical'.format(time.time() - start_time))

	cv = CountVectorizer(min_df=NAME_MIN_DF)
	X_name = cv.fit_transform(asdfpoiu['name'])
	print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

	cv = CountVectorizer()
	X_category = cv.fit_transform(asdfpoiu['category_name'])
	print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

	tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
	                     ngram_range=(1, 3),
	                     stop_words='english')
	X_description = tv.fit_transform(asdfpoiu['item_description'])
	print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

	lb = LabelBinarizer(sparse_output=True)
	X_brand = lb.fit_transform(asdfpoiu['brand_name'])
	print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

	X_dummies = csr_matrix(pd.get_dummies(asdfpoiu[['item_condition_id', 'shipping']],
	                                      sparse=True).values)
	print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

	sparse_asdfpoiu = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
	print('[{}] Finished to create sparse asdfpoiu'.format(time.time() - start_time))

if __name__ == '__main__':
    main()