''' 
Week 4, HW4
CS 584: Applied BioNLP
@author Abeed Sarker
email: abeed.sarker@dbmi.emory.edu

Created: 09/06/2020
***DO NOT REDISTRIBUTE***

Questions are in the file titled 'hw4'
'''

import re
import nltk
from nltk.tokenize import sent_tokenize


# file_path = 'science_text/'


# infile = open(file_path + 'scientificpub2')

# text = infile.read()
# print (text)

# #1. a)
# print (text.lower().find('medicine'))

# #b)
# print (text.lower().count('evidence based medicine'))

# #Note: Lowering the text to ensure matches. Be thoughtful about your preprocessing needs always.
# print ('--------------------')


# #2. a)
# '''
# Match checks for a match only at the beginning of the string, while search checks for a match anywhere in the string
# '''
# #b)
# terms_to_search = ['drug','treatment','system','classifier','evidence','cancer','hypertension']
# for t in terms_to_search:
#     #Note the use of the \b character
#     if re.search(r'\b'+t+r'\b',text):
#         print (t, 'is present in the text')
#     else:
#         print(t, 'is not present in the text')

# #c)
# sentences = sent_tokenize(text)
# print (sentences)
# print (len(sentences))
# for s in sentences:
#     if re.search('clinical.*evidence',s.lower()):
#         print (s)
# #Note: using the .* in regex can be very useful, but you also need to be cautious
# #about how you use it.
# print ('---')
# #d
# '''
# Matches the empty string, but only at the beginning or end of a word. 
# A word is defined as a sequence of word characters. 
# Formally, \\b is defined as the boundary between a \\w and a \\W character (or vice versa), 
# or between \\w and the beginning/end of the string. 
# '''
# #e)
# for s in sentences:
#     if re.search('journal'+r'\b',s.lower()):
#         print (s)

# print ('---')
# #f)
# for s in sentences:
#     if re.search('evidence[ -]based',s):
#         print (s)
# print ('---')
# print('-------------')

#3.a)
symptom_dict = {}
infile = open('COVID-Twitter-Symptom-Lexicon.txt')
for line in infile:
    items = line.split('\t')

    symptom_dict[str.strip(items[-1].lower())] = str.strip(items[1])

#b)

infile = open('posts1.txt')
text = infile.read()
sentences = sent_tokenize(text)
print (sentences)
print ('----')
#matched_tuples will be a list of tuples that will store the match information
#so that it can be processed later by a negation scoping function
matched_tuples = []
#go through each sentence
for s in sentences:
    #go through each symptom expression in the dictionary
    for symptom in symptom_dict.keys():
        #find all matches
        for match in re.finditer(r'\b'+symptom+r'\b',s):
            #Note: uncomment below to see the output
            #print (s, symptom_dict[k],match.group(), match.start(), match.end())
            match_tuple = (s,symptom_dict[symptom],match.group(),match.start(),match.end())
            matched_tuples.append(match_tuple)

def in_scope(neg_end, text,symptom_expression):
    '''
    Function to check if a symptom occurs within the scope of a negation based on some
    pre-defined rules.
    :param neg_end: the end index of the negation expression
    :param text:
    :param symptom_expression:
    :return:
    '''
    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = list(nltk.word_tokenize(text_following_negation))
    # this is the maximum scope of the negation, unless there is a '.' or another negation
    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation),3)])
    #Note: in the above we have to make sure that the text actually contains 3 words after the negation
    #that's why we are using the min function -- it will be the minimum or 3 or whatever number of terms are occurring after
    #the negation. Uncomment the print function to see these texts.
    #print (three_terms_following_negation)
    match_object = re.search(symptom_expression,three_terms_following_negation)
    if match_object:
        period_check = re.search('\.',three_terms_following_negation)
        next_negation = 1000 #starting with a very large number
        #searching for more negations that may be occurring
        for neg in negations:
            # a little simplified search..
            if re.search(neg,text_following_negation):
                index = text_following_negation.find(neg)
                if index<next_negation:
                    next_negation = index
        if period_check:
            #if the period occurs after the symptom expression
            if period_check.start() > match_object.start() and next_negation > match_object.start():
                negated = True
        else:
            negated = True
    return negated

#loading the negation expressions
negations = []
infile = open('neg_trigs.txt')
for line in infile:
    negations.append(str.strip(line))
print (negations)

#now to check if a concept is negated or not
for mt in matched_tuples:
    is_negated = False
    #Note: I broke down the code into simpler chunks for easier understanding..
    text = mt[0]
    cui = mt[1]
    expression = mt[2]
    start = mt[3]
    end = mt[4]
    #uncomment the print calls to separate each text fragment..
    #print('=------=')

    #go through each negation expression
    for neg in negations:
        #check if the negation matches anything in the text of the tuple
        for match in re.finditer(r'\b'+neg+r'\b', text):
            #if there is a negation in the sentence, we need to check
            #if the symptom expression also falls under this expression
            #it's perhaps best to pass this functionality to a function.
            # See the in_scope() function
            is_negated = in_scope(match.end(),text,expression)
            if is_negated:
                print (text,'\t',cui+'-neg')
                break
    if not is_negated:
        print (text,'\t',cui)

    #print('=------=')




