import os
import argparse

import numpy as np
import pandas as pd
import pickle
import time 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

from scipy.sparse import csr_matrix, hstack

from utils import handle_missing_inplace, cutting, to_categorical


parser = argparse.ArgumentParser()

parser.add_argument(
    '--file_path', type=str,
    help = 'The path to the csv/tsv file that contains the set to clean.')

parser.add_argument(
    '--save_path', type=str,
    help = 'The directory to save the cleaned dataset (will be pkl files).')

parser.add_argument(
    '--is_training', type=bool, default = True, 
    help = 'Is this the training set?')


# maybe i can learn how to approach this stuff differently (maybe just make more arguments?)



def main():
    NUM_BRANDS = 4004
    NUM_CATEGORIES = 1001
    NAME_MIN_DF = 10
    MAX_FEATURES_ITEM_DESCRIPTION = 39000

    if FLAGS.file_path.endswith('.tsv'):
        dat = pd.read_table(FLAGS.file_path,engine='c')
    else: 
        dat = pd.read_table(FLAGS.file_path, sep=',', engine = 'python')
        
    start_time = time.time()

    handle_missing_inplace(dat)
    print('[{}] Finished to handle missing'.format(time.time() - start_time))

    cutting(dat)
    print('[{}] Finished to cut'.format(time.time() - start_time))

    to_categorical(dat)
    print('[{}] Finished to convert categorical'.format(time.time() - start_time))

    
    if not FLAGS.is_training:
        with open(FLAGS.save_path + '/cv_name_save.pkl', 'rb') as pickle_in:
            cv_name = pickle_in
        with open(FLAGS.save_path + '/cv_category_save.pkl', 'rb') as pickle_in:
            cv_category = pickle_in
        with open(FLAGS.save_path + '/tv_desc_save.pkl', 'rb') as pickle_in:
            tv_desc = pickle_in
        with open(FLAGS.save_path + '/lb_brand_save.pkl', 'rb') as pickle_in:
            lb_brand = pickle_in
    else:
        cv_name = CountVectorizer(min_df=NAME_MIN_DF)
        cv_name.fit(dat['name'])
        
        cv_category = CountVectorizer()
        cv_category.fit(dat['category_name'])
        
        tv_desc = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
        tv_desc.fit(dat['item_description'])
        
        lb_brand = LabelBinarizer(sparse_output=True)
        lb_brand.fit(dat['brand_name'])
    
    
    X_name = cv_name.transform(dat['name'])
    print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

    X_category = cv_category.transform(dat['category_name'])
    print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

    X_description = tv_desc.transform(dat['item_description'])
    print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

    X_brand = lb_brand.transform(dat['brand_name'])
    print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(dat[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

    sparse_dat = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
    print('[{}] Finished to create sparse dat'.format(time.time() - start_time))

    ## may as well get the pickle for price here
    price_label = dat['price']
    
    if FLAGS.is_training:
        pickle.dump(sparse_dat, open(FLAGS.save_path + '/sparse_mat.pkl','wb'))
        pickle.dump(price_label, open(FLAGS.save_path + '/label.pkl','wb'))
    
    ## save the trained vectorizes and stuff for future cleaning (like with validation set)
    
    if FLAGS.is_training:
        pickle.dump(sparse_dat, open(FLAGS.save_path + '/sparse_mat_val.pkl','wb'))
        pickle.dump(price_label, open(FLAGS.save_path + '/label_val.pkl','wb'))
        pickle.dump(cv_name, open(FLAGS.save_path + '/cv_name_save.pkl','wb'))
        pickle.dump(cv_category, open(FLAGS.save_path + '/cv_category_save.pkl','wb'))
        pickle.dump(tv_desc, open(FLAGS.save_path + '/tv_desc_save.pkl','wb'))
        pickle.dump(lb_brand, open(FLAGS.save_path + '/lb_brand_save.pkl','wb'))
    
    print('Done!')

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    main()