'''
Week 6, HW6
CS 584: Applied BioNLP
@author Thiago Santos
'''

'''
    Answers:

    Due to time and hardware limitations, the search hyperparamters were limited to minimal
    and therefore they may not be optimal for the dataset itself. 
        folds = 3
        SVM Paramters: ['rbf','sigmoid']
        SVM C: 1,2,4,8,16
        ngram(1, 2)
        max_features=2000

    Q1) 
    Best hyperparameters:
        {'svm_classifier__C': 2, 'svm_classifier__kernel': 'rbf'}
        Optimal C found: 2
        Optimal kernel rbf

    (i) unoptimized SVM Performance: 0.34043715846994534
    (ii) optimized SVM Performance:  0.82
    iii naive bayes MultinomialNB Performance:     0.76

    Q2) Best SVM and naive from from question 1 was used. Random forest was created without any hard tunning
        Performance: 0.76

    Q3) Plot can be found in the folder, as training_classifiers_performance.png

'''

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import warnings
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

st = stopwords.words('english')
stemmer = PorterStemmer()


def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df


def preprocess_text(raw_text):

    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))

def grid_search_hyperparam_space(params, pipeline, folds, training_texts, training_classes):
        grid_search = GridSearchCV(estimator=pipeline, param_grid=params, refit=True, cv=folds, return_train_score=False, scoring='accuracy',n_jobs=-1)
        grid_search.fit(training_texts, training_classes)
        return grid_search


def run(size_training = 1):
    #Load the data
    f_path = './data/AirlineSentiment.csv'
    data = loadDataAsDataFrame(f_path)
    data.airline_sentiment = pd.factorize(data.airline_sentiment)[0]
    texts = data['text']
    classes = data['airline_sentiment']
    ids = data['tweet_id']

    # split based on size training
    data_split, classes_split = data, classes 
    texts_split, ids_split = texts, ids
    if size_training <1:
        data_split, _, classes_split, _ = train_test_split(data, classes,test_size=1-size_training,random_state=0)
        texts_split, _, ids_split, _ = train_test_split(texts, ids,test_size=1-size_training,random_state=0)

    # split data
    training_data, test_data, training_classes, test_classes = train_test_split(data_split, classes_split,test_size=0.2,random_state=0)
    training_texts, test_texts, training_ids, test_ids = train_test_split(texts_split, ids_split,test_size=0.2,random_state=0)

    #PREPROCESS THE DATA
    training_texts_preprocessed = [preprocess_text(tr) for tr in training_texts]
    test_texts_preprocessed = [preprocess_text(te) for te in test_texts]


    #VECTORIZER
    vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=2000)


    #CLASSIFIER
    svm_classifier = svm.SVC(gamma='scale')

    #SIMPLE PIPELINE
    pipeline = Pipeline(steps = [('vec',vectorizer),('svm_classifier',svm_classifier)])
    #pipeline ensures vectorization happens in each fold of grid search (you could code the entire process manually for more flexibility)

    grid_params = {
         'svm_classifier__C': [1,2,4,8,16],
         'svm_classifier__kernel': ['rbf','sigmoid'],
    }

    #SEARCH HYPERPARAMETERS
    print("Searching for SVM best params")
    folds = 3
    grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_texts_preprocessed,training_classes)

    print('Best hyperparameters:')
    print(grid.best_params_)
    

    # print('All scores:')
    # all_means = grid.cv_results_['mean_test_score']
    # all_standard_devs = grid.cv_results_['std_test_score']
    # all_params = grid.cv_results_['params']
    # for mean, std, params in zip(all_means, all_standard_devs, all_params ):
    #     print('Mean:',mean, 'Standard deviation:', std, 'Hyperparameters:',  params)

    c_ = grid.best_params_['svm_classifier__C']
    kernel_ = grid.best_params_['svm_classifier__kernel']

    print('Optimal C found:',c_)
    print('Optimal kernel',kernel_)


    # VECTORIZE            
    training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed).toarray()            
    test_data_vectors = vectorizer.transform(test_texts_preprocessed).toarray()


    #CLASSIFY AND EVALUATE -- Q1: 

    # GaussianNK
    clf_naive = MultinomialNB()#GaussianNB()
    clf_naive.fit(training_data_vectors, training_classes) 
    pred_GNB = clf_naive.predict(test_data_vectors)
    acc_GNB = accuracy_score(pred_GNB,test_classes)
    print('Performance on held-out test set for Gaussian NB ... :')
    print(acc_GNB)

    # unoptimized SVM
    #clf = make_pipeline(StandardScaler(), svm.SVC(gamma='scale'))
    clf = svm.SVC(max_iter=20) # Linear Kernel
    clf.fit(training_data_vectors, training_classes) 
    print('Performance on held-out test set for unoptimized SVM ... :')
    print(accuracy_score(clf.predict(test_data_vectors),test_classes))

    
    print('Performance on held-out test set for optimal SVM ... :')
    pred_SVM = grid.predict(test_texts_preprocessed)
    acc_SVM = accuracy_score(pred_SVM,test_classes)
    print(acc_SVM)


    # Q2 

    clf_forest = DecisionTreeClassifier()
    clf_forest.fit(training_data_vectors, training_classes) 
    pred_RF = clf_forest.predict(test_data_vectors).reshape(1,test_data_vectors.shape[0])

    pred_SVM = pred_SVM.reshape(1,test_data_vectors.shape[0])
    pred_GNB = pred_GNB.reshape(1,test_data_vectors.shape[0])

    all_votes = np.concatenate((pred_RF,pred_SVM,pred_GNB), axis=0)

    values, indices = np.unique(all_votes, return_inverse=True)
    majority_vote = values[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(all_votes.shape), None, np.max(indices) + 1), axis=0)]

    acc_ensemble = accuracy_score(majority_vote,test_classes)
    print('Performance on held-out test set for ensemble voting SVM, Random forest and Naive ... :')
    print(acc_ensemble)


    return acc_GNB, acc_SVM, acc_ensemble

if __name__ == '__main__':

    training_size = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    acc_GNB, acc_SVM, acc_ensemble = [], [], []
    for t_size in training_size:
        acc_GNB_, acc_SVM_, acc_ensemble_ = run(size_training = t_size)
        acc_GNB.append(round(acc_GNB_, 3))
        acc_SVM.append(round(acc_SVM_, 3))
        acc_ensemble.append(round(acc_ensemble_, 3))




    labels = ["TRAIN " + str(x) for x in training_size]
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, acc_GNB, width, label='Naive')
    rects2 = ax.bar(x + width/2, acc_SVM, width, label='SVM')
    rects3 = ax.bar(x + width +  width/2, acc_ensemble, width, label='Ensemble')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of models over training sizes')
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
    autolabel(rects3)

    fig.tight_layout()

    plt.show()
        







