# -*- coding: utf-8 -*-
"""
create train and test base data:

    1. User all user order history to get unique uid-pid

    2. add order detail for train and test

    3. add aisle_id + departmentid for feature concat

    4. pickle save
"""

from config.AsConfig import *
import pandas as pd
import numpy as np
import os
import pickle

def main():

    aisles = pd.read_csv(os.path.join(CONST.DATADIR, "aisles.csv"), dtype={'aisle_id': np.uint8, 'aisle': 'category'})
    departments = pd.read_csv(os.path.join(CONST.DATADIR, "departments.csv"),
                              dtype={'department_id': np.uint8, 'department': 'category'})

    products = pd.read_csv(os.path.join(CONST.DATADIR, "products.csv"), dtype={'product_id': np.uint16,
                                                                      'aisle_id': np.uint8,
                                                                      'department_id': np.uint8})

    order_prior = pd.read_csv(os.path.join(CONST.DATADIR, "order_products__prior.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order': np.uint8,
                                                                                      'reordered': bool})

    order_train = pd.read_csv(os.path.join(CONST.DATADIR, "order_products__train.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order': np.uint8,
                                                                                      'reordered': bool})
    orders = pd.read_csv(os.path.join(CONST.DATADIR, "orders.csv"), dtype={'order_id': np.uint32,
                                                                  'user_id': np.uint32,
                                                                  'eval_set': 'category',
                                                                  'order_number': np.uint8,
                                                                  'order_dow': np.uint8,
                                                                  'order_hour_of_day': np.uint8
                                                                  })

    ## get unique uid - pid pair
    tmp = orders.loc[orders['eval_set'] =='prior', ['order_id', 'user_id']]
    tmp = tmp.merge(order_prior.loc[:,['order_id', 'product_id']], on = 'order_id', how = 'left')
    uid_pid = tmp.loc[:,['user_id', 'product_id']].drop_duplicates()
    del tmp

    ## add order attribute on uid-pid pair
    df_train = uid_pid.merge(orders.loc[orders['eval_set'] =='train', [i for i in orders.columns if i !='eval_set']],
                             on = 'user_id', how = 'inner')
    df_test = uid_pid.merge(orders.loc[orders['eval_set'] =='test', [i for i in orders.columns if i !='eval_set']],
                            on = 'user_id', how = 'inner')

    df_train = df_train.merge(products.loc[:, [i for i in products.columns if i != 'product_name']],
                              on = 'product_id', how = 'left')
    df_test = df_test.merge(products.loc[:,[i for i in products.columns if i != 'product_name']],
                            on = 'product_id', how= 'left')

    ## add train label
    df_train = df_train.merge(order_train, on =['order_id', 'product_id'], how ='left')
    del df_train['add_to_cart_order']
    df_train.fillna({'reordered':False}, inplace = True)
    df_train['reordered'] = df_train['reordered'].astype(np.uint8)

    print 'train has {}rows, {} user, {} order, {} product {} reordered'.format(df_train.shape[0],
                                                                        len(df_train.user_id.unique()),
                                                                        len(df_train.order_id.unique()),
                                                                        len(df_train.product_id.unique()),
                                                                        sum(df_train['reordered']) )

    print 'test has {}rows, {} user, {} order, {} product'.format(df_train.shape[0],
                                                                        len(df_train.user_id.unique()),
                                                                        len(df_train.order_id.unique()),
                                                                        len(df_train.product_id.unique()))

    ## save both train and test
    with open(os.path.join(CONST.DATADIR,'df_train.pickle'), 'wb') as f :
        pickle.dump(df_train, f)

    with open(os.path.join(CONST.DATADIR,'df_test.pickle'), 'wb') as f :
        pickle.dump(df_test, f)

if __name__ == '__main__':
     main()