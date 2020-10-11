'''
CS584 BioNLP
Week 5 (THU)

@author: Abeed Sarker

'''


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

stemmer = PorterStemmer()



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
    # stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))


if __name__ == '__main__':
    # Load the data
    f_path = './AirlineSentiment.csv'
    data = loadDataAsDataFrame(f_path)[:1000]
    texts = data['text']
    classes = data['airline_sentiment']
    ids = data['tweet_id']

    # SPLIT THE DATA (we could use sklearn.model_selection.train_test_split)
    training_set_size = int(0.8 * len(data))
    training_data = data[:training_set_size]
    training_texts = texts[:training_set_size]
    training_classes = classes[:training_set_size]
    training_ids = ids[:training_set_size]

    test_data = data[training_set_size:]
    test_texts = texts[training_set_size:]
    test_classes = classes[training_set_size:]
    test_ids = ids[training_set_size:]

    # PREPROCESS THE DATA
    training_texts_preprocessed = [preprocess_text(tr) for tr in training_texts]
    test_texts_preprocessed = [preprocess_text(te) for te in test_texts]
    '''
        #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below   
        training_texts_preprocessed = []
        for tr in training_texts:
            # you can do more with the training text here and generate more features...
            training_texts_preprocessed.append(preprocess_text(tr))
        '''




    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(training_texts_preprocessed,training_classes)
    for c in [1,16,32]:
        for train_index, test_index in skf.split(training_texts_preprocessed,training_classes):

            training_texts_preprocessed_train = map(training_texts_preprocessed.__getitem__, train_index)
            training_texts_preprocessed_dev = map(training_texts_preprocessed.__getitem__,test_index)

            ttp_train, ttp_test = training_classes[train_index], training_classes[test_index]

            # VECTORIZE
            vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=500)
            training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed_train).toarray()
            test_data_vectors = vectorizer.transform(training_texts_preprocessed_dev).toarray()

            # TRAIN THE MODEL
            gnb = GaussianNB()
            svm_classifier = svm.SVC(C=c, cache_size=200,
                                     coef0=0.0, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=True,
                                     random_state=None, shrinking=True, tol=0.001, verbose=False)

            svm_classifier = svm_classifier.fit(training_data_vectors, ttp_train)
            predictions = svm_classifier.predict(test_data_vectors)
            from sklearn.metrics import accuracy_score

            print (accuracy_score(predictions, ttp_test))

        print('----')
