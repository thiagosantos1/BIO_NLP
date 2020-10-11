"""
Week 8, in-class task
CS 584: Applied BioNLP
@author Abeed Sarker
email: abeed.sarker@dbmi.emory.edu

Created: 10/1/2020
***DO NOT REDISTRIBUTE***

"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

st = stopwords.words('english')
stemmer = PorterStemmer()

word_clusters = {}

def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df


def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    # Replace/remove username
    # raw_text = re.sub('(@[A-Za-z0-9\_]+)', '@username_', raw_text)
    #stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))

def create_contingency_table(pred1,pred2,gold):
    contingency_table = [[0,0],
                         [0,0]]
    print(contingency_table)
    for p1,p2,g in zip(pred1,pred2,gold):
        if p1 == g:
            if p1 == p2:
                contingency_table[0][0]+=1
            else:
                contingency_table[0][1]+=1
        else:
            if p1 == p2:
                contingency_table[1][1]+=1
            else:
                contingency_table[1][0]+=1
    return contingency_table

if __name__ == '__main__':
    #Load the data
    f_path = './AirlineSentiment.csv'
    data = loadDataAsDataFrame(f_path)
    texts = data['text']
    classes = data['airline_sentiment']
    ids = data['tweet_id']

    #SPLIT THE DATA (we could use sklearn.model_selection.train_test_split)
    training_set_size = int(0.8*len(data))
    training_data = data[:training_set_size]
    training_texts = texts[:training_set_size]
    training_classes = classes[:training_set_size]
    training_ids = ids[:training_set_size]

    test_data = data[training_set_size:]
    test_texts = texts[training_set_size:]
    test_classes = classes[training_set_size:]
    test_ids = ids[training_set_size:]

    #PREPROCESS THE DATA
    training_texts_preprocessed = []
    test_texts_preprocessed = []

    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    training_texts_preprocessed = []
    for tr in training_texts:
        # you can do more with the training text here and generate more features...
        training_texts_preprocessed.append(preprocess_text(tr))

    for tt in test_texts:
        test_texts_preprocessed.append(preprocess_text(tt))

        #print(getclusterfeatures(tt))
    #VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100)

    training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed).toarray()
    test_data_vectors = vectorizer.transform(test_texts_preprocessed).toarray()

    #TRAIN THE MODEL
    gnb = GaussianNB()
    svm_classifier = svm.SVC(C=2, cache_size=200,
                             coef0=0.0, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=True,
                             random_state=None, shrinking=True, tol=0.001, verbose=False)

    svm_classifier = svm_classifier.fit(training_data_vectors, training_classes)
    gnb_clasifier = gnb.fit(training_data_vectors,training_classes)

    # EVALUATE
    svm_predictions = svm_classifier.predict(test_data_vectors)
    gnb_predictions = gnb_clasifier.predict(test_data_vectors)

    print('SVM Acc:')
    print (accuracy_score(test_classes, svm_predictions))
    print('GNB Acc:')
    print(accuracy_score(test_classes,gnb_predictions))
    print('Confusion matrix, SVM')

    cm_svm = metrics.confusion_matrix(test_classes, svm_predictions)
    print(cm_svm)
    print('Confusion matrix, NB')
    cm_nb = metrics.confusion_matrix(test_classes, gnb_predictions)
    print(cm_nb)
    print('---')

    #WRITE PREDICTIONS TO FILE
    print('Writing predictions to file..')
    outfile = open('./gnb_predictions.txt','w')
    for i,p in zip(test_ids,gnb_predictions):
        outfile.write(str(i)+'\t'+p+'\n')
    outfile.close()
    outfile = open('./svm_predictions.txt','w')
    for i,p in zip(test_ids,svm_predictions):
        outfile.write(str(i) + '\t' + p + '\n')
    outfile.close()
    outfile = open('./gold_classes.txt','w')
    for i,p in zip(test_ids,test_classes):
        outfile.write(str(i) + '\t' + p + '\n')
    outfile.close()


    print('Performing statistical tests...')
    from statsmodels.stats.proportion import proportions_ztest
    from statsmodels.stats.contingency_tables import mcnemar

    print('T-test:')
    num_success_svm = 0
    num_success_nb = 0
    for i in range(0,len(cm_svm)):
        for j in range(0,len(cm_svm)):
            if i == j:
                num_success_svm+=cm_svm[i][j]
                num_success_nb+=cm_nb[i][j]
    print(num_success_svm)
    n_total = len(svm_predictions)


    count = np.array([num_success_svm, num_success_nb])
    nobs = np.array([n_total, n_total])
    stat, pval = proportions_ztest(count, nobs)
    print(pval)
    print('---')

    print('McNemar\'s test:')
    contingency_table = create_contingency_table(svm_predictions,gnb_predictions,test_classes)
    print(contingency_table)
    pval = mcnemar(contingency_table,exact=True)
    print(pval)