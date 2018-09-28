import os
os.getcwd()
os.chdir("/Users/suncicie/Study/Project/DataMining/Competition/DaGuanNLP/DeepLearningforClassification-master/RNNClassification")
os.getcwd()

import xgboost  as xgb
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn import neighbors
from sklearn.ensemble import ExtraTreesClassifier
import string
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
import logging
from collections import Counter
logging.basicConfig(format='%(asctime)s: %(levelnames)s: %(message)s',level=logging.DEBUG)
label = pd.read_csv('data2/train_class.csv')['class']


def train_test(model_name):
    logging.info("start concat  models -------")
    train=[]
    test=[]
    for name in model_name:
        train_tmp = pd.read_csv("data/%s_prb_train.csv" % name)
        test_tmp = pd.read_csv("data/%s_prb_test.csv" % name)
        train.append(train_tmp)
        test.append(test_tmp)
    trainX = pd.concat([i for i in train], axis=1)
    testX = pd.concat([i for i in test], axis=1)
    return trainX,testX

# (train, test, "xgb", is_test=False, is_break=False, is_pandas=True)
def model_stacking(train, test, clf, is_test=True, is_pre=False, is_pandas=False):
'''
is_test: True时只用一折交叉输出结果就行
is_pre:True时 当basemodel 用，保存中间结果
is_break:
'''
    def xgb_Classifier(X_train, y_train, X_val):
        param = {'silent': 1,
                 'eta': 0.1,
                 'max_depth': 4,
                 'subsample': 0.8,
                 'colsample_bytree': 0.5,
                 'objective': 'multi:softprob',
                 'eval_metric': 'mlogloss',
                 'num_class': 19,
                 'seed': 3}
        plst = param.items()
        num_round = 200
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(plst, dtrain, num_round)
        dtest = xgb.DMatrix(X_val)
        y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
        return model, y_pred

    def svc_Classifier(X_train, y_train, X_val,test):
        # 根据类型要改参数
        model = svm.LinearSVC(dual=False)
        model.fit(X_train, y_train)
        y_pred = model.decision_function(X_val)
        test_pre = model.decision_function(test)
        return model, y_pred,test_pre
    def log_Classifier(X_train, y_train, X_val,test):
        # 要改参数
        model= LogisticRegression(C=4,dual=True)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)
        test_pre = model.predict_proba(test)
        return model, y_pred,test_pre

    def knn_Classifier(X_train, y_train, X_val, test):

        model = neighbors.KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)
        test_pre = model.predict_proba(test)
        return model, y_pred, test_pre

    # 可以copy base model 里面的模型
    def run_Classifier(classifier,is_test):
        # log_loss = []
        # acc = []
        pred_train = np.zeros([label.shape[0], 19])
        pred_test = np.zeros([len(test), 19])
        test_pred_result_all = np.zeros([len(test), 19])

        kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=1001)
        for fold_counter, (train_idx, val_idx) in enumerate(kf.split(train, label)):
            logging.info("start %d fold -----------" % fold_counter)
            if is_pandas == False:
                X_train, X_val = train[train_idx], train[val_idx]
            else:
                X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
            y_train, y_val = label[train_idx], label[val_idx]

            clf, y_pred, test_pred = classifier(X_train, y_train, X_val, test)
            # only for prb -------------
            y_pred_result = np.argmax(y_pred, axis=1)
            logging.info("loss %f" % metrics.log_loss(y_val, y_pred_result))
            logging.info("accurary:" % metrics.accuracy_score(y_val, y_pred_result))
            if is_test:
                exit()
            pred_train[val_idx]=y_pred
            test_pred_result_all[, fold_counter]= y_pred_result
            pred_test = pred_test + test_pred

            pred_test=pred_test/5

        logging.info("saving the inner result --------------------")
        res_bag = []
        for r in test_pred_result_all:
            res_bag.append(int(Counter(r).most_common(1)[0][0]))

        res = pd.DataFrame({"class": res_bag, 'id': range(0, len(test), 1)})
        res[["id", "class"]].to_csv('data/res_%s_%s.csv' % (clf_name, data_type), index=None)

        pred_train = pd.DataFrame(pred_train)
        pred_train.columns = ['%s_%s_%s_%s' % (clf_name, data_type, k, str(i)) for i in range(19)]
        pred_train.to_csv('data/%s_%s_train.csv' % (clf_name, data_type), index=False, encoding='utf-8')
        pred_test = pd.DataFrame(pred_test)
        pred_test.columns = ['%s_%s_%s' % (clf_name, data_type, str(i)) for i in range(19)]
        pred_test.to_csv('data/%s_%s_test.csv' % (clf_name, data_type), index=False, encoding='utf-8')

    classifier = eval('%s_Classifier' % clf_name)
    if is_test:
        # 如果是test就只选1折来试一下模型
        run_Classifier(classifier,is_test)
    elif is_pre:
        # 返回train_prob test_prob voting 的 result 且保存
        run_Classifier(classifier, is_test=False)
    else:
        clf, y_pred, test_pred = classifier(train, label, test)
        y_pred = np.argmax(y_pred, axis=1)

        res = pd.DataFrame({"class": y_pred, 'id': range(0, len(test), 1)})
        res[["id", "class"]].to_csv('data/res_ens_%s_%s.csv' % (clf_name, data_type), index=None)

model_name=[]
train,test = train_test(model_name)
label=
model_stacking()