# -*- coding: utf-8 -*-
"""
Get user specific features:
    all has user as unique key
    1. average reorder rate
    2. unique number of product, department, aisle
    3. average/med number of product, department, aisle per order
    4. average/med order dow, order_hour_of_day, days_since_prior_order
"""

from config.AsConfig import *
import pandas as pd
import numpy as np
import os
import pickle
import gc
pd.set_option('display.max.columns', 100 )

def main():

    print 'read in data for feature engineering - user specific'
    with open(os.path.join(CONST.DATADIR, 'orders_detail.pkl'), 'rb') as f:
        orders_detail = pickle.load(f)

    print 'start feature engineering'
    col = 'uid_'

    print 'calculating f0....'
    # reorder rate
    f0 = orders_detail.groupby('user_id').\
        agg({'reordered': lambda x : float(sum(x))/len(x)}).\
        rename({'reordered': 'uid_avg_reorder'})

    # count-product: diversification of user order
    items = ['product_id', 'aisle_id', 'department_id']
    renamedict = dict([(i, 'uid_total_' + i.split('_')[0]) for i in items]) # how to rename feature column

    print 'calculating f1....'
    f1 = orders_detail.groupby('user_id'). \
        agg({'product_id': lambda x: len(x.unique()),
             'aisle_id': lambda x: len(x.unique()),
             'department_id': lambda x: len(x.unique())}).\
        rename(columns = renamedict)

    print 'calculating f2....'
    # avg/med-order-product:
    f2 = orders_detail.groupby(['user_id', 'order_id']). \
        agg({'product_id': lambda x: len(x.unique()),
             'aisle_id': lambda x: len(x.unique()),
             'department_id': lambda x: len(x.unique())}).\
        groupby('user_id').\
        agg({'product_id': ['mean','median'],
             'aisle_id': ['mean','median'],
             'department_id': ['mean','median']})
    f2.columns = [col + j + '_' + i for i, j in f2.columns.to_flat_index()]

    print 'calculating f3....'
    #avg/med time pattern
    f3 = orders_detail.groupby(['user_id', 'order_id']).\
        agg({ 'order_dow': 'mean',
              'order_hour_of_day': 'mean',
              'days_since_prior_order': 'mean'}).\
        groupby('user_id').\
        agg({'order_dow': ['mean', 'median'],
             'order_hour_of_day': ['mean', 'median'],
             'days_since_prior_order': ['mean', 'median']})
    f3.columns = [col + j + '_' + i  for i, j  in f3.columns.to_flat_index()]  #flatten hierarchy columns

    print 'Merge feature together'
    #merge all the features and save
    feature_list = ['f' + str(i) for i in range(4)]
    result = None
    for feature in feature_list:
        if result is None:
            result = eval(feature)
            nrow = result.shape[0]
        else:
            result = result.merge(eval(feature), left_index = True, right_index = True, how = 'left')
            # check if there is duplicate records in the feature
            if result.shape[0]!= nrow:
                raise('There is duplicate records in the features!')
        gc.collect()

    print 'Final feature has shape {}'.format(result.shape)
    print 'Final feature has columne {}'.format(result.columns)

    print 'writing feature to feature_user.pkl'
    with open(os.path.join(CONST.DATADIR, 'feature_user.pkl'), 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()

