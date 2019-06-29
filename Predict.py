"""
Predict Script
    1. Initialize modeler instance
    2. If batch mode, confirm batch files already exist
    3. start batch tranining
"""

import numpy as np
import os
import pickle
import gc
import pandas as pd
pd.set_option('display.max.columns', 100 )
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from config.AsConfig import *
from utils.model_util import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchmode', type = bool, required = True) ## predict in batch/all
    parser.add_argument('--trainstyle', type = str, required = True) ## boosting/bagging model
    parser.add_argument('--modelname', type = str, required = True) ## general model name
    args = parser.parse_args()


    modeler = LGB_TrainAPI_Predict(modelfile = CONST.MODELDIR + args.modelname,
                           datafile = CONST.DATADIR + '/df_train_*.pkl')

    # get data
    if args.batchmode:
        # check with all data file exitst, perpare for generator
        modeler.confirm_batch(mode = 'predict')
    else:
        # if not batchmode, get data directly
        modeler.getdata()

    # get model
    modeler.model_load(True)

    # run predict
    if args.batchmode:
        # run batch predict
        modeler.model_batch_predict(trainstyle = 'bagging')
    else:
        # run normal predict
        modeler.model_predict()

    # aggregrate predict into sample format
    tmp = pd.DataFrame(modeler.result)

    threshold = 0.2

    tmp.to_csv('./submission/submission_0529_v1_raw.csv', index= False)

    submission = tmp.loc[tmp['predict']>threshold, : ]
    submission['product_id'] = submission['product_id'].astype(str)

    submission = submission.groupby('order_id').\
        agg({'product_id': lambda x: ' '.join(x)}).\
        rename(columns = {'product_id':'products'})


    ## merge with sample_submission to fillin na
    sample_submission = pd.read_csv(CONST.DATADIR + '/sample_submission.csv')
    submission = sample_submission.loc[:,['order_id']].merge(submission, on = 'order_id', how = 'left')
    submission.fillna({'products':'None'}, inplace = True)

    submission.to_csv('./submission/submission_0529_v1.csv', index = False)


if __name__ == '__main__':
    main()
