!pip install catboost
!pip install scikit-optimize

import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import *
from threading import Thread
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.base import BaseEstimator, ClassifierMixin

def experiment(model_name, model, params, X_train, y_train, X_test, y_test, i, results):

  # configure the cross-validation procedure
  cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

  # define search
  search = BayesSearchCV(model, params, scoring='accuracy', cv=cv_inner, n_iter=10, refit=True, random_state=1, n_jobs=3)

  # execute search
  result = search.fit(X_train, y_train)

  # get the best performing model fit on the whole training set
  best_model = result.best_estimator_

  # evaluate model on the hold out dataset
  yhat = best_model.predict(X_test)

  # evaluate the model
  acc = accuracy_score(y_test, yhat)
  prec = precision_score(y_test, yhat)
  rec = recall_score(y_test, yhat)
  f1 = f1_score(y_test, yhat)
  mcc = matthews_corrcoef(y_test, yhat)

  # store the result
  results.append([model_name, i, acc, rec, prec, f1, mcc, result.best_score_, result.best_params_])

  # report progress
  print(f"{model_name} {i} > acc={acc:.2f}, est={result.best_score_:.2f}, cfg={result.best_params_}")

# definição dos modelos e parametros
model_params = {
          'lr': {'model': LogisticRegression(),
                'params': {
                          'C': Real(1e-4, 1e4, prior='log-uniform'),
                          'fit_intercept': Categorical([True, False]),
                          'solver': Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])}},

          'knn': {'model': KNeighborsClassifier(),
                  'params': {
                            'n_neighbors': Integer(1, 50),
                            'weights': Categorical(['uniform', 'distance']),
                            'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
                            'p': Integer(1, 5)}},

          'nb': {'model': GaussianNB(),
                'params': {
                          'var_smoothing': Real(1e-10, 1e-1, prior='log-uniform')}},

          'dt': {'model': DecisionTreeClassifier(),
                'params': {
                          'criterion': Categorical(['gini', 'entropy']),
                          'splitter': Categorical(['best', 'random']),
                          'max_depth': Integer(3, 30),
                          'min_samples_split': Integer(2, 10),
                          'min_samples_leaf': Integer(1, 10),
                          'max_features': Real(0.1, 1.0, prior='uniform')}},

          'svm': {'model': SVC(),
                  'params': {
                          'C': Real(2**-5, 2**5, prior='log-uniform'),
                          'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                          'degree': Integer(2, 5),  # Somente relevante para o kernel 'poly'
                          'coef0': Real(0, 1),      # Relevante para os kernels 'poly' e 'sigmoid'
                          'gamma': Real(2**-9, 2**1, prior='log-uniform')}},

          'gpc': {'model': GaussianProcessClassifier(),
                  'params': {
                            'optimizer': Categorical(['fmin_l_bfgs_b', None]),
                            'n_restarts_optimizer': Integer(0, 10),
                            'max_iter_predict': Integer(100, 1000)}},

          'mlp': {'model': MLPClassifier(),
                  'params': {
                            'hidden_layer_sizes': Integer(10,100),
                            'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
                            'solver': Categorical(['lbfgs', 'sgd', 'adam']),
                            'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
                            'learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),
                            'learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform'),
                            'max_iter': Integer(100, 1000)}},

          'ridge': {'model': RidgeClassifier(),
                    'params': {
                              'alpha': Real(1e-4, 1e4, prior='log-uniform'),
                              'fit_intercept': Categorical([True, False]),
                              'solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])}},

          'rf': {'model': RandomForestClassifier(),
                'params': {
                          'n_estimators': Integer(10, 500),
                          'criterion': Categorical(['gini', 'entropy']),
                          'max_depth': Integer(3, 30),
                          'min_samples_split': Integer(2, 10),
                          'min_samples_leaf': Integer(1, 10),
                          'max_features': Real(0.1, 1.0, prior='uniform'),
                          'bootstrap': Categorical([True, False]),
                          'class_weight': Categorical(['balanced', 'balanced_subsample', None])}},

          'qda': {'model': QuadraticDiscriminantAnalysis(),
                  'params': {
                            'reg_param': Real(0, 1, prior='uniform'),
                            'store_covariance': Categorical([True, False]),
                            'tol': Real(1e-5, 1e-1, prior='log-uniform')}},

          'ada': {'model': AdaBoostClassifier(),
                  'params': {
                            'n_estimators': Integer(10, 500),
                            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                            'algorithm': Categorical(['SAMME', 'SAMME.R'])}},

          'gbc': {'model': GradientBoostingClassifier(),
                  'params': {
                            'n_estimators': Integer(10, 500),
                            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                            'max_depth': Integer(3, 10),
                            'min_samples_split': Integer(2, 10),
                            'min_samples_leaf': Integer(1, 10),
                            'max_features': Real(0.1, 1.0, prior='uniform'),
                            'subsample': Real(0.1, 1.0, prior='uniform')}},

          'lda': {'model': LinearDiscriminantAnalysis(),
                  'params': {
                            'solver': Categorical(['svd', 'lsqr', 'eigen']),
                            'shrinkage': Real(0, 1, prior='uniform'),
                            'tol': Real(1e-6, 1e-4, prior='log-uniform')}},

          'et': {'model': ExtraTreesClassifier(),
                'params': {
                          'n_estimators': Integer(10, 500),
                          'criterion': Categorical(['gini', 'entropy']),
                          'max_depth': Integer(3, 30),
                          'min_samples_split': Integer(2, 10),
                          'min_samples_leaf': Integer(1, 10),
                          'max_features': Real(0.1, 1.0, prior='uniform'),
                          'bootstrap': Categorical([True, False]),
                          'class_weight': Categorical(['balanced', 'balanced_subsample', None])}},

          'xgboost': {'model': XGBClassifier(),
                      'params': {
                                'learning_rate': Real(0.01, 0.3, prior='uniform'),
                                'n_estimators': Integer(50, 500),
                                'max_depth': Integer(3, 10),
                                'min_child_weight': Integer(1, 10),
                                'gamma': Real(0, 1, prior='uniform'),
                                'subsample': Real(0.5, 1, prior='uniform'),
                                'colsample_bytree': Real(0.5, 1, prior='uniform'),
                                'reg_alpha': Real(0, 1, prior='uniform'),
                                'reg_lambda': Real(1, 3, prior='uniform'),
                                'scale_pos_weight': Real(1, 5, prior='uniform')}},

          'lightgbm': {'model': LGBMClassifier(),
                      'params': {
                                'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                                'n_estimators': Integer(10, 500),
                                'num_leaves': Integer(2, 100),
                                'max_depth': Integer(3, 10),
                                'min_child_samples': Integer(1, 50),
                                'min_child_weight': Real(1e-5, 1e-3, prior='log-uniform'),
                                'subsample': Real(0.1, 1.0, prior='uniform'),
                                'colsample_bytree': Real(0.1, 1.0, prior='uniform'),
                                'reg_alpha': Real(0, 1, prior='uniform'),
                                'reg_lambda': Real(0, 1, prior='uniform')}},

          'catboost': {'model': CatBoostClassifier(verbose=0),
                      'params': {
                                'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                                'iterations': Integer(10, 500),
                                'depth': Integer(3, 10),
                                'l2_leaf_reg': Real(1, 10, prior='uniform'),
                                'border_count': Integer(1, 255),
                                'bagging_temperature': Real(0, 1, prior='uniform'),
                                'random_strength': Real(1e-9, 10, prior='log-uniform')}}
}

# create dataset
X, y = make_classification(n_samples=1000, n_features=50, random_state=1, n_informative=10, n_redundant=10)

results_file = "exp.xlsx"

# enumerate splits
results = []

# configure the cross-validation procedure
cv_outer = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)

#for train_ix, test_ix in cv_outer.split(X,y):
for i, (train_ix, test_ix) in enumerate(cv_outer.split(X, y)):

  # split data
  X_train, X_test = X[train_ix, :], X[test_ix, :]
  y_train, y_test = y[train_ix], y[test_ix]

  #---------Usado com paralelismo:
  threads = []

  for model_name, mp in model_params.items():

    #Sem paralelismo:
    #experiment(model_name, model_params.get(model_name).get('model'), model_params.get(model_name).get('params'), X_train, y_train, X_test, y_test, i, results)

    #---------Usado com paralelismo:
    exp = Thread(target=experiment,args=[model_name, mp['model'],mp['params'], X_train, y_train, X_test, y_test, i, results])
    exp.start() #inicia thread
    threads.append(exp) #adiciona na lista para salvar a referencia da thread

  #---------Usado com paralelismo:
  for i in range (len(threads)):
    threads[i].join() #retoma o resultado para o programa chamador

# save results to file
df = pd.DataFrame(results, columns=['model', 'run', 'acc', 'rec', 'prec', 'f1', 'mcc', 'best_score', 'best_params'])
df.to_excel(results_file, index=False)