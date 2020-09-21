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
#Perform sentence tokenization
sents = sent_tokenize(filetext)
print ('The number of sentences in the article: '),
print (len(sents))

#Perform word tokenization
words = word_tokenize(filetext)
print ('The total number of words in the article: '),
print (len(words))

print ('Total number of word types in the article: '),
print (len(set(words)))


print ('After preprocessing')
lowercase_words = word_tokenize(filetext.lower())
print (lowercase_words)
stemmed_words = []
for pw in lowercase_words:
    if not pw in sw and not pw in ['.',',']:
        stemmed_words.append(stemmer.stem(pw))
print (stemmed_words)


print ('The total number of words in the article after preprocessing: '),
print (len(stemmed_words))

print ('Total number of word types: '),
print (len(set(stemmed_words)))


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
'''
for k in before_preproc.keys():
    print k,'\t',before_preproc[k]
'''


print 'Top 10 most frequent terms:'
print sorted(before_preproc.items(), key=operator.itemgetter(1),reverse=True)[:10]


after_preproc = term_frequency_distribution(stemmed_words)
print 'Top 10 most frequent terms (after preprocessing):'
print sorted(after_preproc.items(), key=operator.itemgetter(1),reverse=True)[:10]

print 'Lexical diversity of the text: '
print (len(set(stemmed_words))+0.)/len(stemmed_words)



