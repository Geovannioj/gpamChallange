import scipy
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load data GPAM_Challenge dataset 
dataset = pd.read_csv('olist_order_reviews_dataset.csv')

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('review_score').size())

