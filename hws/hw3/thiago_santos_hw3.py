"""
homework 3
CS 584: Applied BioNLP
@author Thiago Santos

Created: 09/08/2020
"""


        ###### Answers #######
'''
    ## Q1
    Jaccard between scientificpub1 & scientificpub2:  0.1592356687898089
    Jaccard between scientificpub2 & scientificpub3:  0.15671641791044777  
    pub2 and pub3 has the higher score and therefore more similar

    Q2.a)
      Must run the program

    Q2.b)
      Cosine Similarity on scientificpub1 & scientificpub2:  0.3205214218820224
'''
        #####   Code   #######


from nltk.tokenize import sent_tokenize, word_tokenize
from os import listdir
from nltk.stem import *
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(threshold=sys.maxsize)

def jaccard(txt1, txt2):
  return len(set(txt1).intersection(set(txt2))) / len(set(txt1).union(set(txt2)))

def clean_txt(txt):
  lowercase_words = word_tokenize(txt.lower())
  sw = stopwords.words('english')
  stemmed_words = []
  stemmer = PorterStemmer()
  for pw in lowercase_words:
      if not pw in sw and not pw in ['.',',', '(', ')', '[', ']']:
        if not pw.isnumeric():
            stemmed_words.append(stemmer.stem(pw))

  return stemmed_words

file_path = 'science_text/'

##Q1:

s_dirlist = listdir(file_path)
terms_per_doc = {} #to store the terms(includind repetiton) for each document, filename as key
for doc in s_dirlist:
  terms_per_doc[doc] = clean_txt(open(file_path + doc).read())

jac_1_2 = jaccard(terms_per_doc['scientificpub1'],terms_per_doc['scientificpub2'])
jac_2_3 = jaccard(terms_per_doc['scientificpub2'],terms_per_doc['scientificpub3'])
print("Jaccard between scientificpub1 & scientificpub2: ", jac_1_2)
print("Jaccard between scientificpub2 & scientificpub3: ", jac_2_3, " pub2 and pub3 has the higher score and therefore more similar")


#TODO: complete the answer...
print ('-------------------------END OF Q1-----------------------')

##Q2 a)

#TODO: complete the answer...
vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=5000)

texts_combined = []
for doc in s_dirlist:
  for key,data in terms_per_doc.items():
    out = ""
    out = ' '.join(map(str, data)) 
    texts_combined.append(out)

matrix = vectorizer.fit_transform(texts_combined).toarray()

# convert to one-hot encod
pub1_vect = np.where(matrix[0] > 0, 1, 0)
pub2_vect = np.where(matrix[1] > 0, 1, 0)
pub3_vect = np.where(matrix[2] > 0, 1, 0)


print("\nVector representation of scientificpub1")
print(pub1_vect)

print("\n\nVector representation of scientificpub2")
print(pub2_vect)

print("\n\nVector representation of scientificpub3")
print(pub3_vect)

#b) cosine similarity
#TODO: compute the cosine similarity
print ('-------------------------END OF Q2-----------------------')
cos_pub1_pub2 = cosine_similarity([pub1_vect],[pub2_vect])[0][0]
print("\nCosine Similarity on scientificpub1 & scientificpub2: ", cos_pub1_pub2,"\n")



