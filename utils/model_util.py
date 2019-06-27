# -*- coding: utf-8 -*-
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
from config.AsConfig import *
import sys
import pickle
from glob import glob
import numpy as np
from tqdm import tqdm

class LGBBasic(object):
    def __init__(self, modelfile, datafile):
        ## main Input to train model
        self.df = None
        self.modelfile = modelfile
        self.datafile = datafile

        ## model parameter
        self.target = CONST.TARGET
        self.params = CONST.PARAMS
        self.test_need = CONST.TEST_NEED
        self.stratify = CONST.STRATIFY
        self.num_iter = CONST.NUM_ITER
        self.early_stopping = CONST.EARLY_STOPPING

        ## Model Related
        self.df_split = None
        self.features = None
        self.model = None
        self.result = None

    def getdata(self):
        with open(self.datafile, 'rb') as f:
            self.df = pickle.load(f)

    def confirm_batch(self, mode):
        if mode =='train':
            files = glob(self.datafile)
            self.nbatch = len(files)
            print 'Data Batch file confirmed {} batches found'.format(self.nbatch)
        elif mode =='predict':
            files = glob(self.modelfile)
            self.nbatch = len(files)
            print 'Model Batch file confirmed {} batches found'.format(self.nbatch)

    def getdata_generator(self):
        files = glob(self.datafile)
        i = 0
        while True:
            with open(files[i], 'rb') as f:
                ibatch = pickle.load(f)
            i = (i+1)%len(files)  # allow read in all train data multiple times
            yield ibatch

    def datasplit(self):
        df = self.df
        # force no shuffle so that each set has the order from same suer
        if self.test_need:
           if self.stratify:
                train_x, oob_x, train_y, oob_y = train_test_split(df, df.loc[:, self.target],
                                                                  test_size=0.4,
                                                                  random_state=CONST.SEED,
                                                                  shuffle=False,
                                                                  stratify = df.loc[:,CONST.STRATIFY_COL ].values)
                test_x, valid_x, test_y, valid_y = train_test_split(oob_x, oob_y,
                                                                    test_size=0.75,
                                                                    random_state=CONST.SEED,
                                                                    shuffle=False,
                                                                    stratify=oob_x.loc[:,CONST.STRATIFY_COL].values)
           else:
               train_x, oob_x, train_y, oob_y = train_test_split(df, df.loc[:, self.target],
                                                                 test_size=0.4,
                                                                 shuffle=False,
                                                                 random_state=CONST.SEED)
               test_x, valid_x, test_y, valid_y = train_test_split(oob_x, oob_y,
                                                                   test_size=0.75,
                                                                   shuffle=False,
                                                                   random_state=CONST.SEED)
           print'Sample Split train shape ={}, valid shape = {}, test shape = {}'.format(train_x.shape,
                                                                                      valid_x.shape,
                                                                                      test_x.shape)
           self.df_split = {'train_x': train_x,
                 'train_y': train_y,
                 'valid_x': valid_x,
                 'valid_y': valid_y,
                 'test_x': test_x,
                 'test_y': test_y}

        else:
            if self.stratify:
                train_x, valid_x, train_y, valid_y = train_test_split(df, df.loc[:, self.target],
                                                                      test_size=0.2,
                                                                      random_state=CONST.SEED,
                                                                      shuffle = False,
                                                                      stratify =df.loc[:, self.target])
            else:
                train_x, valid_x, train_y, valid_y = train_test_split(df, df.loc[:, self.target],
                                                                      test_size=0.2,
                                                                      shuffle=False,
                                                                      random_state=CONST.SEED)

            print 'Sample Split train shape ={}, valid shape = {}'.format(train_x.shape,
                                                                          valid_x.shape)
            self.df_split = {'train_x': train_x,
                             'train_y': train_y,
                             'valid_x': valid_x,
                             'valid_y': valid_y}

    def model_save(self, batchmode):
        if not batchmode:
            print 'model saved at {}'.format(self.modelfile)
            with open(self.modelfile , 'wb') as f:
                pickle.dump(self.model, f)
        else:
            for i, model in self.model_collect.items():
                modelfile = self.modelfile.format(i)

                print 'model saved at {}'.format(modelfile)
                with open(modelfile, 'wb') as f:
                    pickle.dump(model, f)


    def model_load(self, batchmode):
        if not batchmode:
            with open(self.modelfile , 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model_collect = []
            files = sorted(glob(self.modelfile))
            for i in files:
                with open(i, 'rb') as f:
                    self.model_collect.append(pickle.load(f))

    def feature_selection(self, mode):
        if mode =='train ':
            features = self.df.columns
            Unused_feature = CONST.FEATURE_ABANDON + CONST.FEATURE_TARGET
            features = [i for i in features if i not in Unused_feature]
        elif mode =='predict':
            features = self.model_collect[0].feature_name()
        self.features = features
        print 'Feature selection {} of features selected'.format(len(self.features))



class LGB_TrainAPI_Train(LGBBasic):
    def __init__(self, modelfile, datafile):
        super(LGB_TrainAPI_Train, self).__init__(modelfile, datafile)

    def model_run_batch_boosting(self, stage_predict):
        print '='*5 + ' start running batch training ' + '=' *5
        model_collect = {}
        result_collect = {}
        batch = self.getdata_generator()

        for i in range(self.nbatch):
            print 'getting data for batch {}...'.format(i)
            self.df = batch.next()

            if i ==0:
                # in the first batch select features
                self.feature_selection(mode = 'train')
            print 'Splitting data for batch {}...'.format(i)
            self.datasplit()


            print 'Start Model Training for batch {}...'.format(i)
            if i ==0:
                model = None

            model, result = self.train_process(train_x=self.df_split['train_x'].loc[:, self.features],
                                                         train_y=self.df_split['train_y'],
                                                         params=self.params,
                                                         n_estimators=self.num_iter,
                                                         early_stopping=self.early_stopping,
                                                         valid_x=self.df_split['valid_x'].loc[:, self.features],
                                                         valid_y=self.df_split['valid_y'],
                                                         test_x=self.df_split['test_x'].loc[:, self.features],
                                                         stage_predict=stage_predict,
                                                         keep_training_booster= True,
                                                         init_model = model)
            self.result = result
            self.test_eval()
            model_collect[i] = model
            result_collect[i] = result

        self.model_collect = model_collect
        self.result_collect = result_collect
        print '=' * 5 + ' Model Training Finished ' + '=' * 5

    def model_run_batch_bagging(self, stage_predict):
        # run each model on one batch and do ensemble predict on test
        print '='*5 + ' start running batch training ' + '=' *5
        model_collect = {}
        result_collect = {}
        batch = self.getdata_generator()

        for i in range(self.nbatch):
            print 'getting data for batch {}...'.format(i)
            self.df = batch.next()

            if i ==0:
                # in the first batch select features
                self.feature_selection(mode= 'train')
            print 'Splitting data for batch {}...'.format(i)
            self.datasplit()


            print 'Start Model Training for batch {}...'.format(i)
            model, result = self.train_process(train_x=self.df_split['train_x'].loc[:, self.features],
                                                         train_y=self.df_split['train_y'],
                                                         params=self.params,
                                                         n_estimators=self.num_iter,
                                                         early_stopping=self.early_stopping,
                                                         valid_x=self.df_split['valid_x'].loc[:, self.features],
                                                         valid_y=self.df_split['valid_y'],
                                                         test_x=self.df_split['test_x'].loc[:, self.features],
                                                         stage_predict=stage_predict,
                                                         keep_training_booster= False,
                                                         init_model = None)
            self.result = result
            self.test_eval()
            model_collect[i] = model
            result_collect[i] = result

        self.model_collect = model_collect
        self.result_collect = result_collect
        print '=' * 5 + ' Model Training Finished ' + '=' * 5

    def model_run(self, stage_predict):
        self.feature_selection()
        print '='*5 + ' start running batch training ' + '=' *5
        self.model, self.result = self.train_process(train_x=self.df_split['train_x'].loc[:, self.features],
                                                     train_y=self.df_split['train_y'],
                                                     params=self.params,
                                                     n_estimators=self.num_iter,
                                                     early_stopping=self.early_stopping,
                                                     valid_x=self.df_split['valid_x'].loc[:, self.features],
                                                     valid_y=self.df_split['valid_y'],
                                                     test_x=self.df_split['test_x'].loc[:, self.features],
                                                     stage_predict=stage_predict)
        print '=' * 5 + ' Model Training Finished ' + '=' * 5

    @staticmethod
    def train_process(train_x, train_y, params,
                      n_estimators, early_stopping,
                      valid_x, valid_y, test_x=None,
                      init_model=None, keep_training_booster=False,
                      stage_predict=False ):

        params['nthread'] = CONST.NTHREAD
        print 'Model Parameter as Belowed'
        print params
        pred_valid = None
        pred_train = None
        pred_test = None

        lgbtrain = lgb.Dataset(train_x, label=train_y)
        lgbvalid = lgb.Dataset(valid_x, label=valid_y)

        if stage_predict:
            stage_predict = {}
            model = lgb.train(params, lgbtrain, num_boost_round=n_estimators,
                              valid_sets=[lgbvalid, lgbtrain],
                              valid_names=['valid', 'train'],
                              early_stopping_rounds=early_stopping, verbose_eval=CONST.VERBOSE,
                              init_model = init_model, keep_training_booster = keep_training_booster,
                              evals_result=stage_predict)
        else:
            model = lgb.train(params, lgbtrain, num_boost_round=n_estimators,
                              valid_sets=lgbvalid,
                              early_stopping_rounds=early_stopping, verbose_eval=CONST.VERBOSE,
                              init_model = init_model, keep_training_booster = keep_training_booster)

        if test_x is not None:
            pred_test = model.predict(test_x, num_iteration=model.best_iteration)

        pred_train = model.predict(train_x, num_iteration=model.best_iteration)
        pred_valid = model.predict(valid_x, num_iteration=model.best_iteration)

        return model, {'pred_train': pred_train,
                       'pred_valid': pred_valid,
                       'pred_test': pred_test}

    def test_eval(self):
        self.test = pd.DataFrame({'user_id': self.df_split['test_x']['user_id'],
                                  'target': self.df_split['test_y'],
                                  'pred': self.result['pred_test']})
        self.test.sort_values(by = 'user_id', inplace = True)
        # cut row by user
        row_cut = np.arange(0, self.test.shape[0], 1)[~self.test.user_id.duplicated()]

        print 'Evaluating test set with shape {}'.format(self.test.shape)
        threshold_list = np.round(np.arange(0.1, 1, 0.01).tolist(), 2)

        print 'Using below threshold_list {}'.format(threshold_list)
        mean_f1_list = {}
        tmp_old = 0
        for threshold in threshold_list:
             tmp_new = self.mean_f1_score(self.test['pred'] > threshold,
                                                  self.test['target'],
                                                  row_cut)
             print '  calculating mean F1 score for threshold = {}, F1 = {}'.format(threshold, tmp_new)

             if tmp_new < tmp_old:
                 break

             tmp_old = tmp_new
             mean_f1_list[str(threshold)] = tmp_old
        print mean_f1_list
        max_f1 = sorted(mean_f1_list, key = lambda x: x[1], reverse = True)[0]
        print max_f1
        print 'Highest Mean F1 score = {}, threshold = {}'.format(mean_f1_list[max_f1], max_f1)

    @staticmethod
    def mean_f1_score(pred_label, target, row_cut):
        mean_f1 = 0
        n_user = len(row_cut)
        for i in range(n_user -1):
            pred_label_tmp = pred_label.iloc[row_cut[i]:row_cut[i+1]]
            target_tmp = target.iloc[row_cut[i]:row_cut[i+1]]
            mean_f1 += LGB_TrainAPI_Train.f1_score(pred_label_tmp, target_tmp)
        return mean_f1/n_user

    @staticmethod
    def f1_score(pred_label, target):
        if sum(target) == 0:
            # special case when there is no positive value of target
            if sum(pred_label) == 0:
                return 1
            else:
                return 0
        else:
            TP = float(sum(pred_label & target))
            if TP == 0:
                return 0
            precision = TP / (sum(pred_label) + sys.float_info.epsilon) ## predict true
            recall = TP / (sum(target) + sys.float_info.epsilon) ## real true
            f1 = 2 * (precision * recall) / (precision + recall)
        return f1

class LGB_TrainAPI_Predict(LGBBasic):
    def __init__(self,modelfile, datafile):
        super(LGB_TrainAPI_Predict, self).__init__(modelfile, datafile)

    def model_predict(self, trainstyle):
        ## user for predict as a whole
        self.df['target'] = self.model.predict(self.df.loc[:,self.features])

    def model_batch_predict(self, trainstyle):
        print '=' * 5 + ' start running batch prediction ' + '=' * 5
        batch = self.getdata_generator()
        result = {'order_id': [],
                  'product_id':[],
                  'predict':[]}
        for i in range(self.nbatch):

            print 'getting data for batch {}...'.format(i)
            self.df = batch.next()
            if i == 0:
                self.feature_selection(mode = 'predict')

            result['order_id'].extend(self.df['order_id'])
            result['product_id'].extend(self.df['product_id'])

            if trainstyle == 'bagging':
                print 'Bagging predict...'
                predict = [0] * self.df.shape[0] ## Initialize predict

                for j, imodel in enumerate(self.model_collect):
                    print '  Emsemble predict using model {}'.format(j)
                    predict += imodel.predict(self.df.loc[:, self.features])

                predict = predict/len(self.model_collect)

            elif trainstyle == 'boosting':
                print 'Boosting predict ...'
                predict = self.model_collect[-1].predict(self.df.loc[:, self.features])

            result['predict'].extend(predict)
        print 'Predict Finished'
        self.result = result