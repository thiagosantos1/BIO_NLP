from nltk.stem import *
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
import operator
from collections import defaultdict
from gensim.parsing.preprocessing import strip_tags, strip_punctuation,strip_numeric,remove_stopwords

def term_frequency_distribution(termlist):
    frequency_distrib = defaultdict(int)
    for t in termlist:
        frequency_distrib[t]+=1
    return frequency_distrib

def top_k_words(list_file, k=10):
	words = sorted(list_file, key=operator.itemgetter(1),reverse=True)[:10]
	out = ", ".join('\''+x[0]+'\'' for x in words)
	return out

infile = open('science_text/scientificpub1')
filetext = infile.read()

sents = sent_tokenize(filetext)
print("\nQuestion 1")
print("\ta)")
print("\t    length of the document in sentences: ", len(sents))

words = word_tokenize(filetext)
print("\t    length of the document in words: ", len(words))

print('\t    Total number of word types in the article: ', len(set(words)))

print("\n\tb)")

s_lenght = 0
for w in set(words):
	s_lenght += len(w)
print("\t    Average length of an unprocessed token in the file: ",s_lenght/len(set(words)))


lowercase_words = word_tokenize(filetext.lower())
sw = stopwords.words('english')
stemmed_words = []
stemmer = PorterStemmer()
for pw in lowercase_words:
    if not pw in sw and not pw in ['.',',', '(', ')', '[', ']']:
    	if not pw.isnumeric():
        	stemmed_words.append(stemmer.stem(pw))


before_preproc = term_frequency_distribution(words)
after_preproc = term_frequency_distribution(stemmed_words)

print("\n\tc)")
top_10 = top_k_words(before_preproc.items())
print("\t    10 most frequent terms in the document before preprocessing: ", top_10)

print("\n\td)")
top_10 = top_k_words(after_preproc.items())
print("\t    10 most frequent terms in the document after preprocessing: ", top_10)


print("\n\te)")

print("\t    What type of noise is present in the file? Does it affect the statistics we generated?")
print("\t          We have multiple types of noise in the text, including:\n"+
	  "                     * Ponctuation\n" + 
	  "                     * Numbers\n" + 
	  "                     * Fixed structure words (headers)\n" + 
	  "                     * Misspelling words\n" + 
	  "                     * Uppercase words\n" + 
	  "                     * Stop words\n" +
	  "                     * Many others\n" )
print("\t     Does it affect the statistics we generated?  Yes it does. Not only on the statistics but also on\n"+
	  "\t     how we can undestand the data using mathematical models");



print("\nQuestion 2")

infile = open('science_text/scientificpub2')
filetext_2 = infile.read()

infile = open('science_text/scientificpub3')
filetext_3 = infile.read()

corpus = filetext + filetext_2 + filetext_3

print("\ta)")
sents = sent_tokenize(corpus)

sent_1 = sent_tokenize(filetext.lower())
sent_2 = sent_tokenize(filetext_2.lower())
sent_3 = sent_tokenize(filetext_3.lower())
avg_sent = (len(sent_1) + len(sent_2) + len(sent_3) )/3.0
print("\t    length of the document in sentences: ", len(sents))
print("\t        Or --> Average document length(3 docs) for the whole corpus in sentences: ",avg_sent )

words = word_tokenize(corpus)
words_1_b = word_tokenize(filetext)
words_1 = word_tokenize(filetext.lower())
words_2_b = word_tokenize(filetext_2)
words_2 = word_tokenize(filetext_2.lower())
words_3_b = word_tokenize(filetext_3)
words_3 = word_tokenize(filetext_3.lower())
avg_words = (len(words_1) + len(words_2) + len(words_3) )/3.0
print("\t    length of the document in words: ", len(words))
print("\t        Or --> Average document length(3 docs) for the whole corpus in words: ",avg_words )



sw = stopwords.words('english')
stemmed_words_1 = []
stemmed_words_2 = []
stemmed_words_3 = []
stemmer = PorterStemmer()
for pw in words_1:
    if not pw in sw and not pw in ['.',',', '(', ')', '[', ']']:
    	if not pw.isnumeric():
        	stemmed_words_1.append(stemmer.stem(pw))
for pw in words_2:
    if not pw in sw and not pw in ['.',',', '(', ')', '[', ']']:
    	if not pw.isnumeric():
        	stemmed_words_2.append(stemmer.stem(pw))
for pw in words_3:
    if not pw in sw and not pw in ['.',',', '(', ')', '[', ']']:
    	if not pw.isnumeric():
        	stemmed_words_3.append(stemmer.stem(pw))

print("\tb)")
print("\t    Which document in the folder has the highest lexical diversity?")
print("\t         Before cleaning each file from corpus we get the following distribution of unique words")
print("                      * scientificpub1: ", len(set(words_1_b)))
print("                      * scientificpub2: ", len(set(words_2_b)))
print("                      * scientificpub3: ", len(set(words_3_b)))

print("\t         After cleaning each file from corpus we get the following distribution of unique words")
print("                      * scientificpub1: ", len(set(stemmed_words_1)))
print("                      * scientificpub2: ", len(set(stemmed_words_2)))
print("                      * scientificpub3: ", len(set(stemmed_words_3)))

print("\t    This gives us an idea that the file scientificpub3 may have more diversity since it contains a greather\n" +
	  "\t         number of words. However, it's difficult to draw a conclusion, since we aren't sure about\n"+
	  "\t         how spread the words are. Also, the total number of unique words from scientificpub1 is not far\n"+
	  "\t         One of theniques we could apply is clustering, to see up to how many topics each document holds")



