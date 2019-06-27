# -*- coding: utf-8 -*-
"""
Merge data for feature engineering

    merge all detail for prior order
"""

from config.AsConfig import *
import pandas as pd
import numpy as np
import os
import pickle


def main():

    products = pd.read_csv(os.path.join(CONST.DATADIR, "products.csv"), dtype={'product_id': np.uint16,
                                                                               'aisle_id': np.uint8,
                                                                               'department_id': np.uint8})

    order_prior = pd.read_csv(os.path.join(CONST.DATADIR, "order_products__prior.csv"), dtype={'order_id': np.uint32,
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

    tmp = orders.loc[orders['eval_set'] == 'prior',: ]
    tmp = tmp.merge(order_prior, on ='order_id', how = 'left')
    tmp['reordered'] = tmp['reordered'].astype(np.uint8)

    del tmp['eval_set']

    tmp = tmp.merge(products.loc[:, [i for i in products.columns if i!= 'product_name']], on = 'product_id', how = 'left')

    print 'prior order has {} rows, {} user, {} product, {} reorderd'.format(tmp.shape[0],
                                                                             len(tmp.user_id.unique()),
                                                                             len(tmp.product_id.unique()),
                                                                             sum(tmp.reordered))
    with open(os.path.join(CONST.DATADIR, 'orders_detail.pkl'), 'wb') as f:
        pickle.dump(tmp, f)

if __name__ == '__main__':
    main()
