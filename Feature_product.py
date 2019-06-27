# -*- coding: utf-8 -*-
"""
Get product, department, aisle specific features:
    all have product_id as key
    because each product_id has one aisle_id and department_id, so we calculate feature for them also.
    1. average reorder rate: bayesian prior
    2. number of order, user: confidence level for above probability
    3. avg/med order dow, order_hour_of_day, days_since_prior_order
    4. weekday pattern: percentage in each weekday
"""

from config.AsConfig import *
import pandas as pd
import numpy as np
import os
import pickle
import gc
pd.set_option('display.max.columns', 100 )

def main():

    print 'read in data for feature engineering - product specific'
    with open(os.path.join(CONST.DATADIR, 'orders_detail.pkl'), 'rb') as f:
        orders_detail = pickle.load(f)

    # Initial merge dataframe
    result = orders_detail.loc[:, ['product_id', 'aisle_id', 'department_id']].drop_duplicates()
    nrow = result.shape[0]

    print 'start feature engineering'
    items = ['product_id', 'department_id', 'aisle_id']
    for item in items:
        # colname for items
        print 'calculating {} .....'.format(item)
        col = item.split('_')[0][0] + item.split('_')[1] # pid, did, aid

        # average reorder rate
        f0 = orders_detail.groupby(item). \
            agg({'reordered': lambda x: float(sum(x)) / len(x)}). \
            rename(columns = {'reordered': col + '_avg_reorder'})

        # number of order and user
        renamedict = dict([(i, col + '_total_' + i.split('_')[0]) for i in ['order_id', 'user_id']])
        f1 = orders_detail.groupby(item).\
            agg({'order_id': 'size',
                 'user_id': lambda x: len(x.unique())}).\
            rename(columns = renamedict)

        # time pattern
        f2 = orders_detail.groupby(item). \
            agg({'order_dow': 'median',
                 'order_hour_of_day': 'median',
                 'days_since_prior_order': 'median'})
        f2.columns = [col + '_med_' + i for i in f2.columns]

        # merge all the features and save
        feature_list = ['f' + str(i) for i in range(3)]

        # unique
        for feature in feature_list:
            result = result.merge(eval(feature), left_on = item, right_index=True, how='left')
            # check if there is duplicate records in the feature
            if result.shape[0] != nrow:
                raise ('There is duplicate records in the features!')

    print 'Final feature has shape {}'.format(result.shape)
    print 'Final feature has column {}'.format(result.columns)

    print 'writing feature to feature_product.pkl'
    with open(os.path.join(CONST.DATADIR, 'feature_product.pkl'), 'wb') as f:
        pickle.dump(result, f)


    with open('./data/feature_product.pkl', 'rb') as f:
        tmp = pickle.load(f)


if __name__ == '__main__':
    main()
