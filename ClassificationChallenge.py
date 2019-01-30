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

#check data distribution
# dataset.hist()
# plt.show()

#validation dataset

array = dataset.values
# print(array[3,2:5])
X = array[:,0:6]
Y = array[:,4]
validation_size = 0.3
seed = 5

X_train, 
X_validation, 
Y_train, 
Y_validation = model_selection.train_test_split(X,
                                                Y,
                                                test_size=validation_size,
                                                 random_state=seed)

#Test options 
scoring = 'accuracy'
seed = 5

#Check different algorithms

models =[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
