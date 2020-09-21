"""
homework 3
CS 584: Applied BioNLP
@author Thiago Santos

Created: 09/14/2020
"""


        ###### Answers #######
'''
  1)
    a) 4
    b) 4

  2) 
    a) The main difference is that search tries to find a match(first one) of the pattern at any position
        of a giving string. On the other hand, match only checks for a match at the biginning of the string.
    
    b) Terms found:  treatment, system, classifier, evidence
    c) "automated text summarisers that find the best clinical evidence reported in collections of medical 
        literature are of potential benefit for the practice of evidence based medicine (ebm).method

        "we sourced the “clinical inquiries” section of the journal of family practice (jfp) and obtained a sizeable 
        sample of questions and evidence based summaries. the whole annotation process took place between 
        december 2010 and february 2011. during the annotation process the annotators also double-checked 
        the automatically extracted components (clinical inquiry, answer, and evidence grade) and corrected 
        them when necessary."

    d) It basically matches an empty string, but only at the beginning or end of a word
    e)  "we sourced the “clinical inquiries” section of the journal of family practice (jfp) and obtained a 
        sizeable sample of questions and evidence based summaries."
    f) Can get through running code

  3)
    a) Can get through running code
    b) Can get through running code
'''
        #####   Code   #######


from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import *
from nltk.corpus import stopwords
import numpy as np
import re 
import sys
import pandas as pd 

def clean_txt(txt):
  lowercase_words = word_tokenize(txt.lower())
  #sw = stopwords.words('english')
  stemmed_words = ""
  stemmer = PorterStemmer()
  for pw in lowercase_words:
      if not pw in ['.',',', '(', ')', '[', ']']:
        if not pw.isnumeric():
            stemmed_words += " " + stemmer.stem(pw)

  return stemmed_words



np.set_printoptions(threshold=sys.maxsize)

file_path = 'science_text/'


infile = open(file_path + 'scientificpub2')
text = infile.read()
infile.close()
lowercase_sentences = sent_tokenize(text.lower())
all_sentences = ' '.join(map(str, lowercase_sentences)) 

#1. a)
#TODO
print("Number of times does the term 'medicine' occur: " , len(re.findall("medicine", all_sentences)))

#b)
#TODO
print("Number of times does 'evidence based medicine' occur: " , len(re.findall("evidence based medicine", all_sentences)))


#2. a)
'''
#TODO
'''

#b)
terms_to_search = ['drug','treatment','system','classifier','evidence','cancer','hypertension']
#TODO
#print(matched_object.group())
print("Terms found: ", end=' ')
for term in terms_to_search:
  if re.search(term,all_sentences) != None:
    print(term, end=', ')
print()


#c)
#TODO
pattern = re.compile('clinical(.*)evidence')
for sentence in lowercase_sentences:
  if re.search(pattern, sentence) != None:
    print(sentence)


#d
print("\nIt basically matches an empty string, but only at the beginning or end of a word")
'''
#TODO

'''

#e)
#TODO
pattern = re.compile('\\bjournal\\b')
for sentence in lowercase_sentences:
  if re.search(pattern, sentence) != None:
    print(sentence)


print ('\n-----------------------------------------------\n')

#f)
#TODO
pattern = re.compile('evidence based|evidence-based')
for sentence in lowercase_sentences:
  if re.search(pattern, sentence) != None:
    print("\n\tNext sentence: \n", sentence)


print ('\n-----------------------------------------------\n')

#3.a)
#TODO
data = pd.read_csv('COVID.txt', sep='\t', lineterminator='\n',usecols=[0,1,2], names=['definition', 'code', 'symptom'], header=None)
symptom_dict = dict(zip(data.symptom, data.code))
#b)

infile = open('neg_trigs.txt')
neg_trig = infile.read().split("\n")
neg_trig = '|'.join(map(str, neg_trig)) 
infile.close()

infile = open('posts1.txt')
text = infile.read()
sentences = sent_tokenize(text)
infile.close()

print ('\n-----------------------------------------------\n')
print("\t\tSymptoms found")
for sent in sentences:
  sent = sent.strip().rstrip()
  sent = re.sub('\n\n', ' ', sent)
  sent_clean = clean_txt(sent) # clean, for better matching
  for key,value in symptom_dict.items():
    key_clean = clean_txt(key)
    match = re.search(key_clean, sent_clean)
    if match != None:
      print(sent + "\t" + str(value),end='')

      # check for negation --> Could use just re to formulate
      index = match.span()[0]
      prev_3_words = ' '.join(sent_clean[0:index].split()[-3:])
      match_prev = re.search(neg_trig, prev_3_words)
      if  match_prev!=None:
        print("-neg.") # no need to remove
        # remove = " " + match_prev.group() + " "
        # print("To remove: ", remove)
        # sent_clean = re.sub(remove," ",sent_clean,1)
      else:
        print()

      # remove tokens from sentence to don't repeat it again 
      tokens = match.group() 
      to_insert = "," * len(tokens.split()) 
      
      sent_clean = re.sub(tokens,to_insert,sent_clean)
      
      
    
#TODO




