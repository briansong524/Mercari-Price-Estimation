import os
import argparse

import numpy as np
import pandas as pd
import pickle 

from utils import normalize_or_standardize, train_mult_models


parser = argparse.ArgumentParser()

parser.add_argument(
    '--train_sparse_path', type=str,
    help = 'The path to the pkl file that contains the training set.')

parser.add_argument(
    '--label_path', type=str,
    help = 'The path to the pkl file that contains the label.')

parser.add_argument(
    '--label_name', type=str,
    help = 'Name of the label to train on in the training set.') 

parser.add_argument(
    '--norm', type=str, default='standardize',
    help = "Normalize or standardize the label. {'none','standardize','normalize'} (default = 'standardize')")

parser.add_argument(
    '--save_path', type=str,
    help = 'The directory to save the models (will be pkl files).')



def main():
    print('importing data...')
    with open(FLAGS.train_sparse_path,'rb') as pickle_open:
        X_train = pickle.load(pickle_open)
    with open(FLAGS.label_path,'rb') as pickle_open:
        Y_train = pickle.load(pickle_open)

    print('training models...')

    #make pickles
    train_mult_models(X_train, Y_train, save_path = FLAGS.save_path, norm_ = FLAGS.norm)

    print('Done!')





if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    main()