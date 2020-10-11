'''
Week 6, HW6
CS 584: Applied BioNLP
@author Thiago Santos
'''

import nltk
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing


st = stopwords.words('english')
stemmer = PorterStemmer()


def loadDAta(f_path):
    df = pd.read_csv(f_path)
    # return df['Text'].to_numpy(),df['Label'].to_numpy()
    return df['Text'].to_numpy(),df['Sentence'].to_numpy()


def preprocess_text(X):
    X_out = []
    for raw_text in X:

        words = [stemmer.stem(w) for w in raw_text.lower().split()]
        X_out.append(" ".join(words))
    
    return X_out


# convert train and test to bag of words with giving ngrams
def bag_of_words(X_train, X_test, ngrams=(1,1), max_features=1000):
    vectorizer = CountVectorizer(ngram_range=ngrams, max_features=max_features)
    X_train_vectors = vectorizer.fit_transform(X_train).toarray()
    X_test_vectors = vectorizer.transform(X_test).toarray() 

    return X_train_vectors, X_test_vectors


def tf_idf(X_train, X_test, ngrams=(1,1), max_features=1000):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngrams, max_features=max_features)
    vectorizer.fit(X_train)
    X_out_train = vectorizer.transform(X_train).toarray()
    X_out_test = vectorizer.transform(X_test).toarray()
    return X_out_train,X_out_test


def classifier(X_train, X_test, y_train, y_test):
    clf_forest = DecisionTreeClassifier()
    clf_forest.fit(X_train, y_train) 
    predictions = clf_forest.predict(X_test)#.reshape(1,X_test.shape[0])
    f1_macro = f1_score(y_test, predictions, average='macro')
    f1_micro = f1_score(y_test, predictions, average='micro')

    return f1_micro, f1_macro


def plot_test(f1_micro, f1_macro):

    labels = ["Bow no bgm", "Bow (1,2) bgm",  "TFIDF no bgm", "TFIDF (1,2) bgm", "Bow + TFIDF no bgm", "BOW + TFIDF (1,2) bgm"]
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, f1_micro, width, label='F1 Micro')
    rects2 = ax.bar(x + width/2, f1_macro, width, label='F1 Macro')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1')
    ax.set_title('F1 of models over different features created and combined')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    path_data = "./data/"
    X_train, y_train = loadDAta(path_data+"train.csv")
    X_test, y_test = loadDAta(path_data+"test.csv")

    X_train = preprocess_text(X_train)
    X_test = preprocess_text(X_test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_train[y_train > 4] = 4
    y_test[y_test > 4] = 4

    y_test = y_test -1
    y_train = y_train -1

    le = preprocessing.LabelEncoder()
    le.fit(np.append(y_train, y_test))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)


    # bag of words features
        # not bigrams
    bow_train_vectors_no_bigrams,bow_test_vectors_no_bigrams = bag_of_words(X_train,X_test, ngrams=(1,1))
        # with bigrams
    bow_train_vectors_bigrams,bow_test_vectors_bigrams = bag_of_words(X_train,X_test, ngrams=(1,2))


    # TF-IDF vectors
        # not bigrams
    tfidf_train_vectors_no_bigrams,tfidf_test_vectors_no_bigrams = bag_of_words(X_train,X_test, ngrams=(1,1))
        # with (2,2) bigrams
    tfidf_train_vectors_bigrams,tfidf_test_vectors_bigrams = tf_idf(X_train,X_test, ngrams=(1,2))


    #### Classification
    f1_micro, f1_macro = [], [] # micro and macro

    # bag of words Classification
        # no bigrams
    training_data_vectors = bow_train_vectors_no_bigrams
    test_data_vectors     = bow_test_vectors_no_bigrams
    f1_mi, f1_ma = classifier(training_data_vectors, test_data_vectors, y_train, y_test)
    f1_micro.append(round(f1_mi,3))
    f1_macro.append(round(f1_ma,3))

        # with 2,2 bigrams
    training_data_vectors = np.concatenate((bow_train_vectors_bigrams, bow_train_vectors_no_bigrams), axis=1)
    test_data_vectors = np.concatenate((bow_test_vectors_bigrams, bow_test_vectors_no_bigrams), axis=1)
    f1_mi, f1_ma = classifier(training_data_vectors, test_data_vectors, y_train, y_test)
    f1_micro.append(round(f1_mi,3))
    f1_macro.append(round(f1_ma,3))

    # TF-IDF Classification
        # no bigrams
    training_data_vectors = tfidf_train_vectors_no_bigrams
    test_data_vectors     = tfidf_test_vectors_no_bigrams
    f1_mi, f1_ma = classifier(training_data_vectors, test_data_vectors, y_train, y_test)
    f1_micro.append(round(f1_mi,3))
    f1_macro.append(round(f1_ma,3))

         # with 2,2 bigrams
    training_data_vectors = np.concatenate((tfidf_train_vectors_no_bigrams, tfidf_train_vectors_bigrams), axis=1)
    test_data_vectors = np.concatenate((tfidf_test_vectors_no_bigrams, tfidf_test_vectors_bigrams), axis=1)
    f1_mi, f1_ma = classifier(training_data_vectors, test_data_vectors, y_train, y_test)
    f1_micro.append(round(f1_mi,3))
    f1_macro.append(round(f1_ma,3))

    # bow combined with TF-IDF
        # no bigrams
    training_data_vectors = np.concatenate((bow_train_vectors_no_bigrams, tfidf_train_vectors_no_bigrams), axis=1)
    test_data_vectors = np.concatenate((bow_test_vectors_no_bigrams, tfidf_test_vectors_no_bigrams), axis=1)
    f1_mi, f1_ma = classifier(training_data_vectors, test_data_vectors, y_train, y_test)
    f1_micro.append(round(f1_mi,3))
    f1_macro.append(round(f1_ma,3))

        # with 2,2 bigrams
    training_data_vectors = np.concatenate((bow_train_vectors_bigrams, tfidf_train_vectors_bigrams), axis=1)
    test_data_vectors = np.concatenate((bow_test_vectors_bigrams, tfidf_test_vectors_bigrams), axis=1)
    f1_mi, f1_ma = classifier(training_data_vectors, test_data_vectors, y_train, y_test)
    f1_micro.append(round(f1_mi,3))
    f1_macro.append(round(f1_ma,3))


    plot_test(f1_micro,f1_macro)


