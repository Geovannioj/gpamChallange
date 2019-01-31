import scipy
import numpy
import math
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#load data GPAM_Challenge dataset 
dataset = pd.read_csv('olist_order_reviews_dataset.csv')

# dataset.review_score = dataset.review_score.astype(int)
# dataset.review_comment_message = str(dataset.review_comment_message.astype(str)

# print(dataset.shape)
# print(dataset.head(20))
# print(dataset.describe())
# print(dataset.groupby('review_score').size())

#check data distribution
# dataset.hist()
# plt.show()

#validation dataset

array = dataset.values

# split comments column
X = array[:,4] 
X_str

#take out the null instances of comments
for comment in X:
    if isinstance(comment, str):
        X_str.append(comment.lower())
    else: 
        str(comment)
        comment = ""
        X_str.append(comment)

#split rating column
Y = array[:,2]

#amount in percentage of the data to be put into the validation set
validation_size = 0.3

#random seed to split the dataset
seed = 5

# Vectorizer to split text
# vectorizer = TfidfVectorizer(ngram_range=(1,2))
# x_vectorized = vectorizer.fit_transform(X)

# #split the datset between train and validation
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,
#                                                 Y,
#                                                 test_size=validation_size,
#                                                 random_state=seed)

# print("Training set has {} samples.".format(X_train.shape[0]))
# print("Test set has {} samples.".format(X_validation.shape[0]))

# classifierKNN = KNeighborsClassifier(n_neighbors=2)
# classifierKNN.fit(X_train, Y_train)
# prediction = classifierKNN.predict(X_validation)
# print("KNN:", accuracy_score(Y_validation, prediction))
