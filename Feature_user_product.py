# -*- coding: utf-8 -*-
"""
Get user product specific features:
    all has user_id, product_id as unique key
    1. average/min/max reorder rate, time_since_last, order_dow, order_hour_of_day, add_to_cart_order
    2. if user reorder this product, what's the last time they order/reorder this product *

    user_id +  depratment_id, aisle_id
    1. number of order, earliest order time
    2. avg/min/max  reorder rate
    3. time since last: min, mean, max

"""

from config.AsConfig import *
import pandas as pd
import numpy as np
import os
import pickle
import gc
pd.set_option('display.max.columns', 100 )

def main():

    print 'read in data for feature engineering - user & prodcut'
    with open(os.path.join(CONST.DATADIR, 'orders_detail.pkl'), 'rb') as f:
        orders_detail = pickle.load(f)

    col = 'uid_pid'
    # uid, pid: avg, min, max reorder rate
    print 'calculating f0....'
    f0 = orders_detail.groupby(['user_id', 'product_id']).\
        agg({'reordered': lambda x : float(sum(x))/len(x)}).\
        rename({'reordered': col + 'avg_reorder'})

    # number of order of the item
    # position in the cart
    print 'calculating f1....'
    f1 = orders_detail.groupby(['user_id', 'product_id']).\
        agg({'order_id': 'count',
             'order_dow': 'mean',
             'order_hour_of_day': 'mean',
             'days_since_prior_order': ['mean', 'min', 'max'],
             'add_to_cart_order': ['mean', 'min', 'max']})
    f1.columns = [col+ '_' + j + '_' + i for i,j in f1.columns.to_flat_index() ]

    # days since the user last purchased the item
    ## get day gap for each order number
    print 'calculating f2....'
    tmp = orders_detail.loc[:, ['order_number', 'user_id', 'days_since_prior_order'] ].drop_duplicates()
    tmp.sort_values(by = ['user_id', 'order_number'], ascending = [True, False], inplace = True)
    daygap = tmp.groupby(['user_id', 'order_number'])['days_since_prior_order'].\
        apply(lambda x: x.cumsum())
    tmp['daygap'] = daygap
    ## get the latest order number for each product
    tmp2 = orders_detail.groupby(['user_id', 'product_id']).\
        agg({'order_number':'max'})
    tmp2.reset_index(level = [0,1], inplace = True)
    ## join latest order number with the time gap for each order number
    f2 = tmp2.merge(tmp.loc[:,['user_id', 'order_number', 'daygap']], on = ['user_id', 'order_number'], how = 'left')
    f2.set_index(['user_id','product_id'], inplace = True)

    del tmp; del tmp2; gc.collect();

    # merge all uid pid features
    print 'merge feature together'
    result = orders_detail.loc[:,['user_id','product_id', 'department_id', 'aisle_id']].drop_duplicates()
    nrow = result.shape[0]
    feature_list = ['f' + str(i) for i in range(3)]
    for feature in feature_list:
        result = result.merge(eval(feature), left_on = ['user_id', 'product_id'] , right_index = True, how = 'left')
        # check if there is duplicate records in the feature
        if result.shape[0]!= nrow:
            raise('There is duplicate records in the features!')
        gc.collect()

    result.rename(columns = {'order_number' : col + '_max_order_number',
                              'daygap': col + '_max_daygap',
                             'reordered':col + '_avg_reorder'}, inplace =True)
    gc.collect()
    # uid + department_id/aisle_id
    featurepair = [('user_id', 'department_id'), ('user_id', 'aisle_id')]
    for item1, item2 in featurepair:
        print 'calculte for id pair {} and {}'.format(item1, item2)

        col = item1.split('_')[0][0] + item1.split('_')[1] + '_' +  item2.split('_')[0][0] + item2.split('_')[1]
        f0 = orders_detail.groupby([item1, item2]).agg({
            'order_number': 'min',
            'reordered': 'mean',
            'order_id': lambda x: len(x.unique())
        }).rename(columns = {
            'order_number': col + '_min_order_number',
            'reordered': col + '_avg_reorder',
            'order_id': col + '_count_order_id'
        })

        f1 = result.groupby([item1, item2]).agg({
            'uid_pid_avg_reorder': ['min', 'max']
        })
        f1.columns = [col + '_min_reorder', col + '_max_reorder']

        f2 = orders_detail.groupby([item1, item2]).agg({
            'days_since_prior_order': ['min', 'max'],
            'add_to_cart_order': ['min', 'max']
        })

        f2.columns = [col + '_' + col2 + '_' + col1 for col1, col2 in f2.columns.to_flat_index() ]

        print 'merge feature together'
        feature_list = ['f' + str(i) for i in range(3)]
        for feature in feature_list:
            result = result.merge(eval(feature), left_on=[item1, item2], right_index=True, how='left')
            # check if there is duplicate records in the feature
            if result.shape[0] != nrow:
                raise ('There is duplicate records in the features!')
            gc.collect()

    # Replacement items

    # streak number of orders in a row the user has purchased the tiem

    # whether the user alreadu ordered the time today

    print 'Final feature has shape {}'.format(result.shape)
    print 'Final feature has column {}'.format(result.columns)

    print 'writing feature to feature_user_product.pkl'
    with open(os.path.join(CONST.DATADIR, 'feature_user_product.pkl'), 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()

