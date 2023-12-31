!pip install tsai

!pip install sktime

from sklearn.datasets import make_classification
from tsai.all import *

X, y = make_classification(n_samples=1000, n_features=100, random_state=1, n_informative=80, n_redundant=10)

learn = TSClassifier(X, y, metrics=accuracy, arch=XceptionTime, train_metrics=True)
learn.fit(10)

from scipy import interpolate

tck = interpolate.splrep(X[0], y=range(40), s=0)
