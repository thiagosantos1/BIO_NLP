'''
CS 584 
Week 2; Lecture 3
@author Abeed Sarker

'''
sent = 'The price of gas today is $2.12 per gallon. We will be needing twenty-five litres.'

from nltk.tokenize import sent_tokenize

sent_tokenized = sent_tokenize(sent)
print (sent)
print (sent_tokenized)
print('Number of sentences:',len(sent_tokenized))


from nltk.tokenize import word_tokenize

term_tokenized = word_tokenize(sent)
print (sent)

print (term_tokenized)
print('Number of terms:', len(term_tokenized))
print('Number of unique terms (types):',len(set(term_tokenized)))