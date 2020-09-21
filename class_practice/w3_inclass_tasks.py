"""
Week 3, Lab work solutions (in class)
CS 584: Applied BioNLP
@author Abeed Sarker
email: abeed.sarker@dbmi.emory.edu

Created: 08/29/2020
***DO NOT REDISTRIBUTE***
"""

from nltk.corpus import stopwords 
import nltk

sw = stopwords.words('english')

def clean(txt):
	lowercase_words = word_tokenize(txt.lower())
	sw = stopwords.words('english')
	stemmed_words = []
	stemmer = PorterStemmer()
	for pw in lowercase_words:
	    if not pw in sw and not pw in ['.',',', '(', ')', '[', ']']:
	    	if not pw.isnumeric():
	        	stemmed_words.append(stemmer.stem(pw))
	return stemmed_words


def jaccard(txt1, txt2):
	return len(set(txt1).intersection(set(txt2))) / len(set(txt1).union(set(txt2)))

print('---Exploring the brown corpus..')
print ('In-class Q1: What are the top 20 most frequently used terms in the brown corpus (without preprocessing..)?')
#Loading the brown corpus

# nltk.download('brown')
from nltk.corpus import brown
words = brown.words()

#Corpus statistics..
print ('Total number of words in the corpus')
print (len(words))
print (list(words[:100]))

#Generating frequency distribution using nltk.FreqDist
words = [w.lower() for w in words]
fd = nltk.FreqDist(words)

#Now we need to apply a sort function to find the most frequent ones...
import operator
sorted_fd = sorted(fd.items(),  key = operator.itemgetter(1), reverse=True)

## TODO  ....


print('--------------------------------------------')

print ('In-class Q2: Generate n-grams for the sentence: "Contiguous sequence of n items from a given sequence?"')
print('---N-grams...')

#N-grams
from nltk import bigrams,trigrams,ngrams,tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
text = 'Contiguous sequence of n items from a given sequence'

tokens = word_tokenize(text)

bigrams_txt = list(bigrams(tokens))

print(bigrams_txt) 



## TODO ....

print ('In-class Q3: Can you do the same for the news texts? Pre-process the text this time. ')
#Reading the text files..
from nltk.stem import *

news1 = open('news_text/sportsnews').read()
news2 = open('news_text/sportsnews2').read()

news1 = clean(news1)
news2 = clean(news2)

bigrams_news1= list(bigrams(news1))
bigrams_news2= list(bigrams(news2))
print(bigrams_news1)
## TODO ....


print('In-class Q4: Compute the jaccard similarity between the unigrams of the two texts?')
print(jaccard(news1, news2))

## TODO ....