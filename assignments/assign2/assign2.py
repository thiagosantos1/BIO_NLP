'''
Week 6, Assignment 2
CS 584: Applied BioNLP
@author Thiago Santos
'''


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier 
import warnings
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import xgboost
from sklearn.model_selection import train_test_split

import itertools

class Parkinson:
  def __init__(self,  path = "data/", run_times = 10,
                data_keep = ['age', 'female', 'fall_location', 'word_count','char_count',
                'sentence_count', 'avg_word_length', 'avg_sentence_lenght']):

    self.path = path
    self.data_keep = data_keep
    self.run_times = run_times
    self.initialize()

  def initialize(self):
    warnings.filterwarnings("ignore")
    self.data = pd.read_csv(self.path + 'pdfalls.csv')
    self.data['fall_class'] = self.data["fall_class"].apply(lambda x: int(x=='Other'))
    scaler = MinMaxScaler()
    self.data['age'] = scaler.fit_transform(self.data["age"].values.reshape(-1,1))

    # more Feature Engineering 
    self.data ['word_count'] = self.data["fall_description"].apply(lambda x: len(str(x).split(" ")))
    self.data['char_count'] = self.data["fall_description"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
    self.data['sentence_count'] = self.data["fall_description"].apply(lambda x: len(str(x).split(".")))
    self.data['avg_word_length'] = self.data['char_count'] / self.data['word_count']
    self.data['avg_sentence_lenght'] = self.data['word_count'] / self.data['sentence_count']
    
    self.data ['word_count'] = scaler.fit_transform(self.data["word_count"].values.reshape(-1,1))
    self.data ['char_count'] = scaler.fit_transform(self.data["char_count"].values.reshape(-1,1))
    self.data ['sentence_count'] = scaler.fit_transform(self.data["sentence_count"].values.reshape(-1,1))
    self.data ['avg_word_length'] = scaler.fit_transform(self.data["avg_word_length"].values.reshape(-1,1))
    self.data ['avg_sentence_lenght'] = scaler.fit_transform(self.data["avg_sentence_lenght"].values.reshape(-1,1))

    self.data_train = self.data.sample(frac = 0.8)
    self.data_test=self.data.drop(self.data_train.index)


    # get labels
    self.y_train  = self.data_train['fall_class'].to_numpy()

    self.y_test = self.data_test['fall_class'].to_numpy()
    

    age_train = self.data_train['age'].to_numpy().reshape(-1,1)
    age_test = self.data_test['age'].to_numpy().reshape(-1,1)

    female = self.data_train['female'].to_numpy().reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    female_train = enc.fit_transform(female).toarray() # dummy variable

    female = self.data_test['female'].to_numpy().reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    female_test = enc.fit_transform(female).toarray() # dummy variable

    fall_location_train = self.data_train['fall_location'].to_numpy()
    fall_location_test = self.data_test['fall_location'].to_numpy()
    
    fall_location_train_clean = self.preprocess_text(fall_location_train)
    fall_location_test_clean = self.preprocess_text(fall_location_test)
    fall_train, fall_test  = self.tf_idf(fall_location_train_clean,fall_location_test_clean,max_features=100)

    fall_description_train = self.data_train['fall_description'].to_numpy()
    fall_description_test = self.data_test['fall_description'].to_numpy()

    fall_description_train_clean = self.preprocess_text(fall_description_train)
    fall_description_test_clean = self.preprocess_text(fall_description_test)
    self.X_train, self.X_test  = self.tf_idf(fall_description_train_clean,fall_description_test_clean)

    # combine all data that is set
    if 'age' in self.data_keep:
      self.X_train = np.concatenate((self.X_train, age_train), axis=1)
      self.X_test = np.concatenate((self.X_test, age_test), axis=1)
    if 'female' in self.data_keep:
      self.X_train = np.concatenate((self.X_train, female_train), axis=1)
      self.X_test = np.concatenate((self.X_test, female_test), axis=1)
    if 'fall_location' in self.data_keep:
      self.X_train = np.concatenate((self.X_train, fall_train), axis=1)
      self.X_test = np.concatenate((self.X_test, fall_test), axis=1)

    if 'word_count' in self.data_keep:
      word_count_train = self.data_train['word_count'].to_numpy().reshape(-1,1)
      word_count_test = self.data_test['word_count'].to_numpy().reshape(-1,1)
      self.X_train = np.concatenate((self.X_train, word_count_train), axis=1)
      self.X_test = np.concatenate((self.X_test, word_count_test), axis=1)

    if 'char_count' in self.data_keep:
      char_count_train = self.data_train['char_count'].to_numpy().reshape(-1,1)
      char_count_test = self.data_test['char_count'].to_numpy().reshape(-1,1)
      self.X_train = np.concatenate((self.X_train, char_count_train), axis=1)
      self.X_test = np.concatenate((self.X_test, char_count_test), axis=1)

    if 'sentence_count' in self.data_keep:
      sentence_count_train = self.data_train['sentence_count'].to_numpy().reshape(-1,1)
      sentence_count_test = self.data_test['sentence_count'].to_numpy().reshape(-1,1)
      self.X_train = np.concatenate((self.X_train, sentence_count_train), axis=1)
      self.X_test = np.concatenate((self.X_test, sentence_count_test), axis=1)

    if 'avg_word_length' in self.data_keep:
      avg_word_length_train = self.data_train['avg_word_length'].to_numpy().reshape(-1,1)
      avg_word_length_test = self.data_test['avg_word_length'].to_numpy().reshape(-1,1)
      self.X_train = np.concatenate((self.X_train, avg_word_length_train), axis=1)
      self.X_test = np.concatenate((self.X_test, avg_word_length_test), axis=1)

    if 'avg_sentence_lenght' in self.data_keep:
      avg_sentence_lenght_train = self.data_train['avg_sentence_lenght'].to_numpy().reshape(-1,1)
      avg_sentence_lenght_test = self.data_test['avg_sentence_lenght'].to_numpy().reshape(-1,1)
      self.X_train = np.concatenate((self.X_train, avg_sentence_lenght_train), axis=1)
      self.X_test = np.concatenate((self.X_test, avg_sentence_lenght_test), axis=1)


  def preprocess_text(self,X):
    sw = stopwords.words('english')
    stemmer = PorterStemmer()
    X_out = []
    for raw_text in X:
      words = [stemmer.stem(w) for w in raw_text.lower().split() if not w in sw and len(w) >3 and w not in ['“',']','\'', '.','com','[',',','.',';',':', '?','!','@', '#', '$','%','&','*','(',')'] and not w.replace('.','',1).isdigit()]
      X_out.append(" ".join(words))
    
    return X_out


  def tf_idf(self,X_train, X_test, ngrams=(1,2), max_features=1000):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngrams, max_features=max_features)
    self.vectorizer = vectorizer
    vectorizer.fit(X_train)
    X_out_train = vectorizer.transform(X_train).toarray()
    X_out_test = vectorizer.transform(X_test).toarray()
    return X_out_train,X_out_test


  def grid_search_hyperparam_space(self,params, pipeline, folds=5):
    grid_search = GridSearchCV(estimator=pipeline, param_grid=params, refit=True, cv=folds, return_train_score=False, scoring='accuracy',n_jobs=-1)
    grid_search.fit(self.X_train, self.y_train)
    return grid_search

  def plot_confusion_matrix(self,cm, classes=['CoM', 'Other'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.close()
    #Add Normalization Option
    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


  def classifier_no_tuning(self, model, model_name='naive bayes', plot=True):

    model.fit(self.X_train, self.y_train) 
    acc,f1_macro,f1_micro = 0,0,0
    for x in range(self.run_times):
      pred = model.predict(self.X_test)
      acc += accuracy_score(pred,self.y_test)
      f1_macro += f1_score(self.y_test, pred, average='macro')
      f1_micro += f1_score(self.y_test, pred, average='micro')

    acc /= self.run_times
    f1_macro /= self.run_times
    f1_micro /= self.run_times
    cm= confusion_matrix(self.y_test, pred )
    print("\n\tAverage(10runs) Performance of {}\nAccuracy: {}\nF1 Macro: {}\nF1 Micro: {}\n".format(model_name, acc,f1_macro,f1_micro))

    if plot:
      self.plot_confusion_matrix(np.array([[20,0],[2,1]]))
    return acc, f1_macro, f1_micro



  def classifier(self, model, grid_params, model_name = 'SVM', plot=True):
    #SEARCH HYPERPARAMETERS
    print("Searching for " + model_name + " best params")
    grid = self.grid_search_hyperparam_space(grid_params,model)
    print('Best hyperparameters:')
    print(grid.best_params_)

    print('Performance on held-out test set for optimal model ... :')
    acc,f1_macro,f1_micro = 0,0,0
    for x in range(self.run_times):
      pred = grid.predict(self.X_test)
      acc += accuracy_score(pred,self.y_test)
      f1_macro += f1_score(self.y_test, pred, average='macro')
      f1_micro += f1_score(self.y_test, pred, average='micro')

    acc /= self.run_times
    f1_macro /= self.run_times
    f1_micro /= self.run_times
    cm= confusion_matrix(self.y_test, pred )
    print("\n\tAverage(10runs) Performance of {}\nAccuracy: {}\nF1 Macro: {}\nF1 Micro: {}\n".format(model_name, acc,f1_macro,f1_micro))
    
    if plot:
      self.plot_confusion_matrix(cm)

    return acc, f1_macro, f1_micro



  def models(self):

    # naive
    acc, f1_macro, f1_micro = self.classifier_no_tuning(MultinomialNB())


    # SVM
    svm_classifier = svm.SVC(gamma='scale')
    grid_params = {
         'C': [1,2,4,8,16,32],
         'kernel': ['rbf','sigmoid',  'poly'],
    }

    acc, f1_macro, f1_micro = self.classifier(svm_classifier,grid_params)

    # decision tree
    clf_forest = DecisionTreeClassifier()
    grid_params = {
         'criterion': ['gini','entropy'],
         'splitter': ['best', 'random'],
         'max_features': ['auto', 'sqrt', 'log2'],
    }
    acc, f1_macro, f1_micro = self.classifier(clf_forest,grid_params, model_name='Decision Tree')


    # Random Forest
    clf_forest = RandomForestClassifier()
    grid_params = {
         'n_estimators': [32,64,100,128,256],
         'criterion': ['gini','entropy'],
         'max_features': ['auto', 'sqrt', 'log2'],
         'bootstrap': [True, False],
    }
    acc, f1_macro, f1_micro = self.classifier(clf_forest,grid_params, model_name='Random Forest')


    # Bagging
    clf_bag = BaggingClassifier()
    acc, f1_macro, f1_micro = self.classifier(clf_forest,grid_params, model_name='Bagging')


    # XGBOST
    acc, f1_macro, f1_micro = self.classifier_no_tuning(xgboost.XGBClassifier(), model_name='XGBOST')


  def run_small_train(self):
    model = svm.SVC(gamma='scale', C=4)
    training_size = [0.4, 0.6, 0.8, 0.99]
    acc, f1_micro, f1_macro = [], [], []
    for size in training_size:
      data_split, _, classes_split, _ = train_test_split(self.X_train, self.y_train,test_size=1-size)
      model.fit(data_split, classes_split)
      pred = model.predict(self.X_test)
      acc.append(round(accuracy_score(pred,self.y_test),2))
      f1_macro.append(round(f1_score(self.y_test, pred, average='macro'),2))
      f1_micro.append(round(f1_score(self.y_test, pred, average='micro'),2))



    labels = ["TRAIN " + str(x) for x in training_size]
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, acc, width, label='Acc')
    rects2 = ax.bar(x + width/2, f1_micro, width, label='F1 Micro')
    rects3 = ax.bar(x + width +  width/2, f1_macro, width, label='F1 Macro')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Performance')
    ax.set_title('Accuracy x F1 Micro x F1 Macro of models over training sizes')
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

  def feature_selection(self):
    model = svm.SVC(gamma='scale', C=4)
    data_keep = ['','age', 'female', 'fall_location', 'word_count','char_count',
                'sentence_count', 'avg_word_length', 'avg_sentence_lenght']

    acc, f1_micro, f1_macro = [], [], []
    data_to_use = []
    for data in data_keep:
      data_to_use.append(data)
      self.data_keep = data_to_use
      self.initialize()
      model.fit(self.X_train, self.y_train)
      pred = model.predict(self.X_test)
      acc.append(round(accuracy_score(pred,self.y_test),2))
      f1_macro.append(round(f1_score(self.y_test, pred, average='macro'),2))
      f1_micro.append(round(f1_score(self.y_test, pred, average='micro'),2))



    labels = ["Features: "  + str(x+1) for x in range(len(data_keep))]
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, acc, width, label='Acc')
    rects2 = ax.bar(x + width/2, f1_micro, width, label='F1 Micro')
    rects3 = ax.bar(x + width +  width/2, f1_macro, width, label='F1 Macro')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Performance')
    ax.set_title('Accuracy x F1 Micro x F1 Macro of models over training sizes')
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

  def run(self):
    self.models()

if __name__ == '__main__':
  obj = Parkinson()
  obj.feature_selection()
  






