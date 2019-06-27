"""
Training Script
    1. Initialize modeler instance
    2. If batch mode, confirm batch files already exist
    3. start batch tranining[Boosting or Bagging] - params need adjust
    4. save model
TODO
    1. dynameic learning rate
    2. dynamic early stopping
    3. init_score at the real probability
    4. optimize for user specific mean f1 score
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
    parser.add_argument('--batchmode', type = bool, required = True) ## train/test
    parser.add_argument('--trainstyle', type = str, required = False) ## bagging/boosting
    parser.add_argument('--modelname', type = str, required = True)
    args = parser.parse_args()

    modeler = LGB_TrainAPI_Train(modelfile = CONST.MODELDIR + '/lgb_bagging_beta_v{}.pkl',
                           datafile = CONST.DATADIR + '/df_train_*.pkl')
    if args.batchmode:
        modeler.confirm_batch(mode = 'train')

        if args.trainstyle =='boosting':
            modeler.model_run_batch_boosting(stage_predict = True)

        elif args.trainstyle =='bagging':
            modeler.model_run_batch_bagging(stage_predict = True)

    else:
        modeler.model_run()

    modeler.model_save(args.batchmode)




if __name__ =='__main__':
    main()
