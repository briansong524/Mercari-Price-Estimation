import os
import argparse

import numpy as np
import pandas as pd
import glob
import pickle 

from utils import optimal_weights_ensemble, rmse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--valid_sparse_path', type=str,
    help = 'The path to the pkl file that contains the validation set.')

parser.add_argument(
    '--label_path', type=str,
    help = 'The path to the pkl file that contains the validation label.')

parser.add_argument(
    '--model_path', type=str,
    help = 'The path to the directory that contains all the models.')

parser.add_argument(
    '--save_path', type=str,
    help = 'The directory to save the models (will be pkl files).')

parser.add_argument(
    '--error', type=str,
    help = 'RMSE or RMSLE')

parser.add_argument(
    '--norm', type=str, default='standardize',
    help = "Normalize or standardize the label. {'none','standardize','normalize'} (default = 'standardize')")


# need parser for if previous stuff was normalized/standardize/none

def main():
    print('importing data...')

    with open(FLAGS.valid_sparse_path,'rb') as pickle_open:
        X_train = pickle.load(pickle_open)
    with open(FLAGS.label_path,'rb') as pickle_open:
        Y_train = pickle.load(pickle_open)
    with open(FLAGS.save_path + '/meanstd.pkl','rb') as pickle_open:
        meanstd = pickle.load(pickle_open)
        mean_standard = meanstd[0]
        std_standard = meanstd[1]
    print('importing models...')

    os.chdir(FLAGS.model_path)
    list_of_models = glob.glob('model_*.pkl')
    num_models = len(list_of_models)
    preds_array = np.zeros((X_train.shape[0], num_models))

    # get the predicted outputs from each model and store in a numpy array 

    for i in range(num_models):
        model_str = list_of_models[i]
        with open(model_str,'rb') as pickle_open:
            model_ = pickle.load(pickle_open)
        prediction = model_.predict(X_train)
        prediction = prediction * std_standard + mean_standard
        preds_array[:,i] = prediction

    #
    if FLAGS.error == 'RMSLE':
        Y_train = np.log1p(Y_train)


    best_w = optimal_weights_ensemble(preds_array,Y_train)
    best_pred = np.matmul(preds_array, best_w)
    best_rmse = rmse(np.squeeze(best_pred), Y_train)

    if FLAGS.norm == "normalize":
        best_pred = np.expm1(best_pred)
        
    print('best weights in order of combined regressors: ' + str(best_w))
    print(' ')
    if FLAGS.error == "RMSE":
    print('RMSE of the ensemble: ' + str(best_rmse))
    else:
    print('RMSLE of the ensemble: ' + str(best_rmse))

    pickle.dump(best_pred, open(FLAGS.save_path + '/ensemble_predictions.pkl','wb'))

    print('Done!')





if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    main()