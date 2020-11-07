from nltk.corpus import brown
import nltk
# nltk.download('universal_tagset')
#nltk.download('treebank')
from sklearn.model_selection import train_test_split


from nltk.tag import UnigramTagger
from nltk.tag import DefaultTagger
from nltk.tag import BigramTagger

from sklearn_crfsuite import metrics as crf_metrics
import sklearn_crfsuite

tagged_sentences = brown.tagged_sents(tagset='universal')


'''
Questions 1 -  Split the dataset into two: 80% for training; 20% for test.
'''
all_sentences = brown.sents()
len_sent = len(all_sentences)
X_train = tagged_sentences[:int(len(tagged_sentences) * 0.8)]
X_test = tagged_sentences[int(len(tagged_sentences) * 0.8):]

'''
Question 2 - Performance of 0.13, 0.9 and 0.91
'''

# using only the default - NN - 0.1308
default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.evaluate(tagged_sentences))


# Unigrams - 0.902
unigram_tagger = UnigramTagger(X_train)
print(unigram_tagger.evaluate(X_test))

# Bigrams with backoff of unigrams - 0.911
bigram_tagger = BigramTagger(X_train, backoff=unigram_tagger)
print(bigram_tagger.evaluate(X_test)) 


'''
Question 3 Performace of 0.77 and 0.79
'''
treebank_tagged_sents = nltk.corpus.treebank.tagged_sents(tagset='universal')
print(default_tagger.evaluate(treebank_tagged_sents))
print(unigram_tagger.evaluate(treebank_tagged_sents)) # 0.77
print(bigram_tagger.evaluate(treebank_tagged_sents))  # 0.79


'''
Question 4-5 - F1 of 0.972 for brown dataset. Better performance
'''

# modified code
def word2features(sent, i):
  word = sent[i][0]
  features = {
    'bias': 1.0,
    'word.lower()': word.lower(),
    'word[-3:]': word[-3:],
    'word[-2:]': word[-2:],
    'word.isupper()': word.isupper(),
    'word.istitle()': word.istitle(),
    'word.isdigit()': word.isdigit(),
  }
  if i > 0:
    word1 = sent[i-1][0]
    features.update({
        '-1:word.lower()': word1.lower(),
        '-1:word.istitle()': word1.istitle(),
        '-1:word.isupper()': word1.isupper(),

    })
  else:
    features['BOS'] = True

  if i < len(sent)-1:
    word1 = sent[i+1][0]
    features.update({
        '+1:word.lower()': word1.lower(),
        '+1:word.istitle()': word1.istitle(),
        '+1:word.isupper()': word1.isupper(),

    })
  else:
    features['EOS'] = True

  return features

def sent2features(sent):
  return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
  return [label for word,label in sent]

def sent2tokens(sent):
  return [token for token,label in sent]


X_train_crf = [sent2features(s) for s in X_train]
y_train_crf = [sent2labels(s) for s in X_train]

X_test_crf = [sent2features(s) for s in X_test]
y_test_crf = [sent2labels(s) for s in X_test]


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train_crf, y_train_crf)


y_pred = crf.predict(X_test_crf)
f1= crf_metrics.flat_f1_score(y_test_crf, y_pred,
                      average='weighted', labels=list(crf.classes_))

print(f1)




