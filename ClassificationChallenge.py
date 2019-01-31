import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#load data GPAM_Challenge dataset 
dataset = pd.read_csv('olist_order_reviews_dataset.csv') 

#take the blank comments row out
dataset = dataset.dropna(subset=['review_comment_message'])

print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())
print(dataset.groupby('review_score').size())

#check data distribution
dataset.hist()
plt.show()

#validation dataset

array = dataset.values

# split comments column
X = array[:,4] 

#split rating column
Y = array[:,2]
Y = Y.astype(int)

#amount in percentage of the data to be put into the validation set
validation_size = 0.3

#random seed to split the dataset
seed = 5

# Vectorizer to split text
vectorizer = TfidfVectorizer(ngram_range=(1,2))
x_vectorized = vectorizer.fit_transform(X)

#split the datset between train and validation
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x_vectorized,
                                                Y,
                                                test_size=validation_size,
                                                random_state=seed)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Test set has {} samples.".format(X_validation.shape[0]))

#Classifier
linearClassificer = LinearSVC()
linearClassificer.fit(X_train, Y_train)
linearPrediction = linearClassificer.predict(X_validation)
print("Accuracy")
print("LinearSVC:",accuracy_score(Y_validation,linearPrediction))

classifierKNN = KNeighborsClassifier(n_neighbors=2)
classifierKNN.fit(X_train, Y_train)
predictionKNN = classifierKNN.predict(X_validation)
print("KNN:", accuracy_score(Y_validation, predictionKNN))


multinomialClassifier = MultinomialNB()
multinomialClassifier.fit(X_train, Y_train)
predictionMUB = multinomialClassifier.predict(X_validation)
print("MultinomialNB:",accuracy_score(Y_validation, predictionMUB))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]),decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Confusion Matrix to the classifiers
confusionMatrixLSVC = confusion_matrix(Y_validation, linearPrediction)
confusionMatrixKNN = confusion_matrix(Y_validation, predictionKNN)
confusionMatrixMUB = confusion_matrix(Y_validation, predictionMUB)

#Plotting
classes = ['1', '2', '3', '4', '5']
plt.figure()

plot_confusion_matrix(confusionMatrixLSVC,
                      classes,
                      normalize=False,
                      title='Confusion Matrix - SVM')
plt.show()
plot_confusion_matrix(confusionMatrixKNN,
                    classes,
                    normalize=False,
                    title="Confusion Matrix - KNN")
plt.show()

plot_confusion_matrix(confusionMatrixMUB,
                    classes,
                    normalize=False,
                    title="ConfusionMatrix - MultinomialNB")
plt.show()