class CONST:
    ## FEATURE SELECTION
    FEATURE_ABANDON = []
    ## use target encoding, insteaad of id itself
    FEATURE_TARGET = ['user_id', 'order_id', 'eval_set', 'reordered']
    TARGET = 'reordered'

    ## MODEL TRAINING PARAMETER
    APPAGE_SPLIT_THRESHOLD = 30
    VERBOSE = 1
    PARAMS = {'objective': 'binary',
            'metric':'cross_entropy',
            'learning_rate': 0.05,
            'bagging_fraction' : 0.8,
            'feature_fraction': 0.8,
            'max_depth': 6,
            'num_leaves': 32}
    NUM_ITER = 1000
    EARLY_STOPPING = 30
    TEST_NEED = True
    NTHREAD = -1
    STRATIFY = False
    STRATIFY_COL = 'user_id'
    SEED = 1234
    ##
    DATADIR = './data'
    MODELDIR = './model'