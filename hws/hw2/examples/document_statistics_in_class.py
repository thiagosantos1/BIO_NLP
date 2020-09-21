'''
CS 584 
Week 2; Lecture 3
@author Abeed Sarker

'''
from nltk.stem import *
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import operator
from collections import defaultdict


sw = stopwords.words('english')
stemmer = PorterStemmer()

infile = open('./news_text/sportsnews2')
#infile = open('./science_text/scientificpub1')
filetext = infile.read().decode('utf8')
print (filetext)

print ('Before preprocessing...')
#Perform sentence tokenization here


print ('The number of sentences in the article: '),

#Perform word tokenization here

print ('The total number of words in the article: '),


print ('Total number of word types in the article: '),


print ('After preprocessing')


print ('The total number of words in the article after preprocessing: '),


print ('Total number of word types: '),


def term_frequency_distribution(termlist):
    '''

    :param termlist: a list of tokens from a document
    :return: the frequency distribution of the tokens
    '''
    frequency_distrib = defaultdict(int)
    for t in termlist:
        frequency_distrib[t]+=1
    return frequency_distrib

before_preproc = term_frequency_distribution(words)


print 'Top 10 most frequent terms:'


after_preproc = term_frequency_distribution(stemmed_words)

print 'Lexical diversity of the text: '
