'''
Week 5, HW5
CS 584: Applied BioNLP
@author Thiago Santos
'''

from nltk import word_tokenize
from nltk.stem import *
from nltk.corpus import stopwords
from collections import defaultdict
from nltk import bigrams
from nltk import ConditionalFreqDist, FreqDist
import re
from scipy import spatial
sw = stopwords.words('english')
stemmer = PorterStemmer()

##Q1
import os


##Homweork
#a) I changed a little to don't repeat code
def make_inverted_index(corpus_path):
    '''
    :param corpus_path: the path to a corpus (which is basically a directory of text files)
    :return: an inverted index as a dictionary; the inverted index should be a defaultdict
        with an n-gram as the key and all the documents that contain the n-gram as a set (or list)
        Also returns the vocab of the set, and also the clean & tokenized file
    '''
    dirlist = os.listdir(corpus_path)
    all_science_text = []
    all_bigrams = []
    science_texts_clean = [] # gonan keep it separate
    inverted_index = defaultdict(set)
    for filename in dirlist:
        print(filename)
        txt = open(corpus_path+filename).read().lower()
        txt = re.sub("\\d", " ", txt).strip()
        filetext = word_tokenize(txt)
        preprocessed_filetext = [stemmer.stem(w) for w in filetext if not w in sw and len(w) >3 and w not in ['“',']','\'', '.','com','[',',','.',';',':', '?','!','@', '#', '$','%','&','*','(',')'] and not w.replace('.','',1).isdigit()]
        #generate the bigrams
        preprocessed_bigrams = [x[0] + "_" + x[1] for x in list(bigrams(preprocessed_filetext))]
        #combine the bigrams with unigrams
        all_bigrams+=preprocessed_bigrams #needed for q2
        combined_ngrams = preprocessed_filetext+preprocessed_bigrams
        all_science_text += combined_ngrams
        science_texts_clean.append(combined_ngrams)
        words = set(combined_ngrams)
        for word in words:

            if word in inverted_index:
                inverted_index[word]+=1
            else:
                inverted_index[word] = 1

    science_vocabulary = set(all_science_text)

    return inverted_index, science_texts_clean, science_vocabulary

##Functions to compute TF-IDF (in-class task..)
def tfs(document, vocab_terms):
    '''
    :param document: a document in a corpus as a string
    :param terms: the terms in the vocabulary
    :return: a list containing counts of each term in the vocabulary
    '''
    # don't need anymore, sice document is already clean and steamed
    # words = [stemmer.stem(w) for w in word_tokenize(document.lower())if not w in sw and len(w) >3 and w not in ['“',']','\'', '.','com','[',',','.',';',':', '?','!','@', '#', '$','%','&','*','(',')'] and not w.replace('.','',1).isdigit()]
    # bigram = list(bigrams(words))
    # combined_ngrams = words+bigram
    tfvec = [document.count(term) for term in vocab_terms]
    return tfvec

#b)
def dfs(inverted_index, vocab_terms):
    '''
    :param inverted_index: an inverted index of the corpus as a dictionary
    :param vocab_terms: the terms in the vocabulary
    :return: a list of document frequencies for each term
    '''
    #TODO
    dfvec = [x for key,x in inverted_index.items() if key in vocab_terms]
    return dfvec

from math import log

def tfidf(numdocs, inverted_index, document, vocab_terms):
    '''
    :param numdocs: the number of documents in the corpus
    :param inverted_index: an inverted index as a dictionary
    :param document: the document as a string, whose
    :param vocab_terms:
    :return:
    '''
    Tfs = tfs(document, vocab_terms)
    Dfs = dfs(inverted_index, vocab_terms)
    N = numdocs
    return [(Tf * log(N / (Df + 1e-100))) for Tf,Df in zip(Tfs, Dfs)]

#c) TODO
inverted_index, science_texts_clean, vocab = make_inverted_index('science_text/')
num_docs = len(science_texts_clean)
tf_idf_vectors = []
for doc in science_texts_clean:
    tf_idf_vectors.append(tfidf(num_docs, inverted_index, doc, vocab))


''' 
    Similarities:
    Seems like Document 1 and 3 is more similar
'''
print("Cosine of TF-IDF from scientificpub1 and scientificpub2: ", 1 - spatial.distance.cosine(tf_idf_vectors[2], tf_idf_vectors[0]) )
print("Cosine of TF-IDF from scientificpub1 and scientificpub3: ", 1 - spatial.distance.cosine(tf_idf_vectors[2], tf_idf_vectors[1]) )
print("Cosine of TF-IDF from scientificpub2 and scientificpub3: ", 1 - spatial.distance.cosine(tf_idf_vectors[0], tf_idf_vectors[1]) )








