import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

NUM_BRANDS = 4004
NUM_CATEGORIES = 1001
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 39000


def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

## blending algorithms    
    

def normalize_or_standardize(dat, norm_ = 'standardize'):
    if norm_ == 'normalize':
        dat = np.log1p(dat)
    mean_norm = np.mean(dat)
    std_norm = np.std(dat) 
    out_ = (dat - mean_norm)/std_norm 
    
    return out_, mean_norm, std_norm
    

def train_mult_models(X_train, Y_train, save_path, norm_ = 'standardize'):
    '''
    Make sure Y_train and Y_test arent scaled. If it is though, maybe it'll be fine lol.
    '''

    # Assertion 
    set_norm_terms = set(('none','standardize','normalize'))
    try:
        assert norm_ in set_norm_terms
    except:
        print('Possible typo: The parameter "norm_" must be string "none", "standardize", or "normalize".')
    
    # normalize
    if norm_ != 'none':
        Y_train, mean_train, std_train = normalize_or_standardize(Y_train, norm_ = norm_)  
        
    # lgb regression
    print('training Light GBM...')
    lgb_dat = lgb.Dataset(X_train, label = Y_train)
    params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
        }

    model_lgb = lgb.train(params, train_set = lgb_dat, num_boost_round = 3000, verbose_eval = 100)

    # ridge regression 
    print('training Ridge Regression...')
    model_ridge = Ridge(solver="sag", fit_intercept=True)
    model_ridge.fit(X_train, Y_train)

    # pickle dump
    pickle.dump(model_lgb, open(save_path + '/model_lgb.pkl','wb'))
    pickle.dump(model_ridge, open(save_path + '/model_ridge.pkl','wb'))
    meanstd = (mean_train, std_train)
    pickle.dump(meanstd, open(save_path + '/meanstd.pkl','wb'))
    
## post-prediction functions

def rmsle(h, y): 
    log_h = np.log(h+1) # the +1 is to prevent 0 
    log_y = np.log(y+1) # writing these to prevent memoryerror
    sq_logs = np.square(log_h - log_y)
    score_ = np.sqrt(np.mean(sq_logs))
    return score_

def rmse(h,y):
    sq_logs = np.square(h-y)
    score_ = np.sqrt(np.mean(sq_logs))
    return score_

def unwind(preds, mean_,std_, norm_ = True):
    unstandardized = preds*std_ + mean_
    if norm_ == True: # norm_ is if the original value (like the label) was normalized with np.logm1
        unstandardized = np.expm1(unstandardized)
    return unstandardized

def optimal_weights_ensemble(Xs, y):
    # make sure Xs is all predictions where each column is predictions by each model
    # y just has to be a vector of true values
    # note: the values of Xs need to be scaled to the values that would go into RMSE
    
    y = np.reshape(y,(-1,1)) # to make it shape [n,1]
    
    first = np.matmul(np.transpose(y),Xs)
    second = np.matmul(np.transpose(Xs), Xs)
    w_ = np.matmul(first, np.linalg.inv(second)) # this should be of size [1,n_weights]
    return np.transpose(w_) #returns [n_weight,1] for easier matrix multiplication 



