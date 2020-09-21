''' 
CS 584
Week 2; Lecture 3
@author Abeed Sarker

'''

##########PREPROCESSING EXAMPLES#############

from nltk.stem import *
stemmer = PorterStemmer()


words_with_morphological_affixes = ['caresses', 'flies', 'dies', 'mules', 'denied',
                                    'died', 'agreed', 'owned', 'humbled', 'sized',
                                    'meeting', 'stating', 'siezing', 'itemization',
                                    'sensational', 'traditional', 'reference', 'colonizer', 'plotted']

print('Checkout these porter stemmer examples...')
for w in words_with_morphological_affixes:
    print (w,'->',stemmer.stem(w))

################STOPWORDS EXAMPLE#################
from nltk.corpus import stopwords
print (set(stopwords.words('english')))

text1 = ['I','like','to','paint']
text2 = ['I', 'Like', 'Painting']
text3 = ['I', 'like', 'to', 'play']

print ('Lowercasing the texts in list 1')
for t in text2:
    print (t.lower())

print('----')
stemmed_text1,stemmed_text2,stemmed_text3 = [],[],[]

print('Now to do some stemming...')
#Can you only keep the 'non-stopwords'?
for t in text1:
    stemmed = stemmer.stem(t)
    stemmed_text1.append(stemmed.lower())
print (stemmed_text1)
for t in text2:
    stemmed = stemmer.stem(t)
    stemmed_text2.append(stemmed.lower())
print (stemmed_text2)
for t in text3:
    stemmed = stemmer.stem(t)
    stemmed_text3.append(stemmed.lower())
print (stemmed_text3)

