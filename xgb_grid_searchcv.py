# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:35:51 2018

@author: Shayak
"""

####XGB####
# 1. XGB, Supply train and dev sets both
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

def xgb_(X_train = X_train, Y_train = Y_train,params = {
             "objective"        :['multi:softmax']
            ,"max_depth"        :[2]
            ,'eta'              :[0.1]
        },X_eval = X_dev, Y_eval = Y_dev):

    
    ######### Apply xgb
    #d_train = xgb.DMatrix(X_train , label = Y_train)
    #d_eval =  xgb.DMatrix(X_eval , label = Y_eval)
    print("# Tuning hyper-parameters for accuracy" )
    print()
    model_xgb = xgb.XGBClassifier()
    clf = GridSearchCV(model_xgb, param_grid = params,cv =3, scoring ="accuracy")
    clf.fit(X_train, Y_train)
    print("\nBest parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_eval, clf.predict(X_eval)
    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))
    print()
    model_xgb = clf.best_estimator_
    xgb.plot_importance(model_xgb)
    
    #Eval model
    Y_dev_pred = model_xgb.predict(X_eval)
    score = accuracy_score(Y_eval,Y_dev_pred)

    return model_xgb,score,clf

xgb_params = {
     'learning_rate'    :[0.1]
    ,'reg_lambda'       :[2,20]
    ,"max_depth"        :[2]
    ,'silent'           :[False]
    ,'n_estimators'     :[200]
    ,'colsample_bytree' :[0.75]
    ,'nthread'          :[3]
    ,'subsample'        :[0.75]
    ,'objective'        :["multi:softmax"]
}


model_xgb,score_xgb,xgb_gridsearch = xgb_(X_train = X_train,X_eval = X_dev,params = xgb_params)
