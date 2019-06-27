# -*- coding: utf-8 -*-
"""
Merge feature into train / test data and save
    1. feature_user.pkl
    2. feature_product.pkl
    3. feature_user_product.pkl
"""


from config.AsConfig import *
import pandas as pd
import numpy as np
import os
import pickle
import gc
pd.set_option('display.max.columns', 100 )
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, required = True) ## train/test
    parser.add_argument('--batch', type = int, required = True)
    args = parser.parse_args()

    # read in train and test data set
    print 'read in {} data'.format(args.dataset)
    if args.dataset == 'train':
        with open(CONST.DATADIR + '/df_train.pkl', 'rb') as f:
            df = pickle.load(f)
        df.sort_values(by = 'user_id', inplace = True) # we want to evaluate on same userid
    elif args.dataset == 'test':
        with open(CONST.DATADIR + '/df_test.pkl', 'rb') as f:
            df = pickle.load(f)
    else:
        raise('Only train and test argument are accepted')

    # merge in batch or not: if the data is too big, merge nrows by rows
    print 'save data in {} batches'.format(args.batch)
    if args.batch >1:
        batchsize = df.shape[0]// args.batch
    else:
        batchsize = df.shape[0]

    # read in all the features
    print 'read in all the features'
    with open(CONST.DATADIR + '/feature_product.pkl', 'rb') as f:
        f_product = pickle.load(f)

    with open(CONST.DATADIR + '/feature_user.pkl' , 'rb') as f:
        f_user = pickle.load(f)

    with open(CONST.DATADIR + '/feature_user_product.pkl', 'rb') as f:
        f_user_product = pickle.load(f)

    # merge feature with train & test data
    # because we user user + product history to predict future: we'd better put same user in the same batch
    for i in range(args.batch):
        print 'start merge feature for batch {}'.format(i)
        if i == args.batch - 1:
            ibatch = df.iloc[i * batchsize: , :]
        else:
            ibatch = df.iloc[i*batchsize:(i+1)*batchsize,: ]

        ibatch = ibatch.merge(f_product, left_on = ['product_id', 'aisle_id', 'department_id'],
                                        right_on = ['product_id', 'aisle_id', 'department_id'], how = 'left')
        ibatch = ibatch.merge(f_user, left_on = ['user_id'], right_index = True, how = 'left')
        ibatch = ibatch.merge(f_user_product, left_on = ['user_id', 'product_id', 'aisle_id', 'department_id'],
                                              right_on = ['user_id', 'product_id', 'aisle_id', 'department_id'], how = 'left')
        print ibatch.isnull().sum()
        ## add time to last order
        ibatch['uid_pid_max_daygap'] = ibatch['uid_pid_max_daygap'] + ibatch['days_since_prior_order']

        gc.collect()
        print 'ibatch shape = {}'.format(ibatch.shape)
        print 'Dumping file...'

        if args.dataset == 'train':
            with open(CONST.DATADIR + '/df_train_batch{}.pkl'.format(i), 'wb') as f:
                pickle.dump(ibatch, f)
        elif args.dataset == 'test':
            with open(CONST.DATADIR + '/df_test_batch{}.pkl'.format(i), 'wb') as f:
                pickle.dump(ibatch, f)


if __name__ == '__main__':
    main()
