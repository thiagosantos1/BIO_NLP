from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import *
from nltk.corpus import stopwords
import numpy as np
import re 
import sys
import pandas as pd 
from nltk.corpus import stopwords
import argparse


class Sympton_detection:
  def __init__(self,  path = "data/", type_neg = "simple", clean_type = 'lower_stop_steam'):
    '''
      type_neg:
        simple: Means my onw solution for negation catch
        professor: Means using the negation solution provided by professor

      clean_type:
        no_cleaning: no cleaning
        lower_case : using only lower case
        lower_stop : lower case and stop words
        lower_steam : lower case and steaming
        lower_stop_steam : lower case, stop words and steaming

    '''
    self.type_neg = type_neg
    self.path = path
    self.clean_type = clean_type
    self.save_pred_out = self.path + clean_type + "_" + type_neg + "_" +  "prediction.xlsx"

    self.initialize()


  def initialize(self):
    infile = open(self.path + 'neg_trigs.txt')
    neg_trig = infile.read().split("\n")
    infile.close()
    self.neg_trig = '|'.join(map(str, neg_trig)) 

    # read COVID Lexicon
    data = pd.read_csv(self.path + 'COVID.txt', sep='\t', lineterminator='\n',usecols=[0,1,2], names=['definition', 'code', 'symptom'], header=None)
    self.symptom_dict = dict(zip(data.symptom, data.code))

    # read golden data, for test
    data = pd.read_excel(self.path + 'Assignment1GoldStandardSet.xlsx')
    text = data['TEXT'].values
    text_id  = data['ID'].values
    self.eval = {} # key as text ID and value as txt

    # feed dictionary
    for index, data in enumerate(text):
      self.eval[text_id[index]] = self.clean_txt(data)

    # check evaluation set
    # for key, value in self.test.items():
    #   print(key, value)
    #   print("\n---------------------------------------------\n")


  def clean_txt(self,txt):

    lowercase_words = word_tokenize(txt)

    sw = stopwords.words('english')
    cleaned = ""
    stemmer = PorterStemmer()
    for pw in lowercase_words:
      if not self.clean_type == 'no_cleaning':
        pw = pw.lower()
        if not pw in [',', '(', ')', '[', ']']:
          if not pw.replace('.','',1).isdigit():
            if self.clean_type == 'lower_stop' or self.clean_type == 'lower_stop_steam':
              if not pw in sw: # stop words
                if self.clean_type == 'lower_stop_steam':
                  cleaned += " " + stemmer.stem(pw)
                else:
                  cleaned += " " + pw
            elif self.clean_type == 'lower_steam':
              cleaned += " " + stemmer.stem(pw)

            elif self.clean_type == 'lower_case':
              cleaned += " " + pw

      else:
        cleaned += " " + pw


    return cleaned


  def my_own_negation(self):

    predictions = {} # will keep predictions. Keys = text ID, value = list --> [0] sysmptom CUIS [1] --> negation flag

    for key_txt, text in self.eval.items():
      sentences = sent_tokenize(text)
      for sent in sentences:
        sent = sent.strip().rstrip()
        sent = re.sub('\n\n', ' ', sent)
        sent_orig = sent
        for key_symp,value in self.symptom_dict.items():
          key_clean = self.clean_txt(key_symp)
          match = re.search(key_clean, sent)
          if match != None:
            # print(sent_orig + "\t" + str(value) )
            symp = "$$$" + str(value)  + "$$$"
            neg  =  "$$$" + str(0) +"$$$"

            # check for negation --> Could use just re to formulate
            index = match.span()[0]
            prev_3_words = ' '.join(sent[0:index].split()[-3:])
            #print(key_symp, " ", prev_3_words +"\n\n")
            match_prev = re.search(self.neg_trig, prev_3_words)
            if  match_prev!=None:
              # print("-neg.") # no need to remove
              neg  =  "$$$" + str(1) +"$$$"

            # remove tokens from sentence to don't repeat it again 
            tokens = match.group() 
            to_insert = "," * len(tokens.split()) 
            
            sent = re.sub(tokens,to_insert,sent)

            
            if not key_txt in predictions:
              predictions[key_txt] = [symp,neg]
            else:
              prev_pred = predictions[key_txt]
              predictions[key_txt] = [ prev_pred[0] + symp, prev_pred[1] + neg] 
      

    self.predictions = predictions

  def negation_professor_solution(self):
    predictions = {} 
    for key_txt, text in self.eval.items():
      sentences = sent_tokenize(text)
      matched_tuples = []
      for s in sentences:
        for symptom in self.symptom_dict.keys():
          for match in re.finditer(r'\b'+symptom+r'\b',s):
            match_tuple = (s,self.symptom_dict[symptom],match.group(),match.start(),match.end())
            matched_tuples.append(match_tuple)

      for mt in matched_tuples:
        is_negated = False
        text = mt[0]
        cui = mt[1]
        expression = mt[2]
        start = mt[3]
        end = mt[4]
        symp_out = "$$$" + str(cui)  + "$$$"
        neg_out  =  "$$$" + str(0) +"$$$" ""
        negations = []
        infile = open(self.path + 'neg_trigs.txt')
        for line in infile:
          negations.append(str.strip(line))

        for neg in negations:
            for match in re.finditer(r'\b'+neg+r'\b', text):
                is_negated = self.in_scope(match.end(),text,expression)
                if is_negated:
                    neg_out  =  "$$$" + str(1) +"$$$"
                    break

        if not key_txt in predictions:
          predictions[key_txt] = [symp_out,neg_out]
        else:
          prev_pred = predictions[key_txt]
          predictions[key_txt] = [ prev_pred[0] + symp_out, prev_pred[1] + neg_out] 
        

    self.predictions = predictions


  # professor's solution
  def in_scope(self,neg_end, text,symptom_expression):

    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = list(word_tokenize(text_following_negation))
    
    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation),3)])

    match_object = re.search(symptom_expression,three_terms_following_negation)
    if match_object:
        period_check = re.search('\\.',three_terms_following_negation)
        next_negation = 1000 
        negations = []
        infile = open(self.path + 'neg_trigs.txt')
        for line in infile:
          negations.append(str.strip(line))
          
        for neg in negations:
            if re.search(neg,text_following_negation):
                index = text_following_negation.find(neg)
                if index<next_negation:
                    next_negation = index
        if period_check:
            if period_check.start() > match_object.start() and next_negation > match_object.start():
                negated = True
        else:
            negated = True
    return negated


  def save_pred(self):
    text_ids, symp, neg = [], [], []

    for text_id, value in self.predictions.items():
      text_ids.append(text_id)
      symp.append(value[0])
      neg.append(value[1])

    df = pd.DataFrame(list(zip(text_ids, symp, neg)), 
               columns =['ID', 'Symptom CUIs', 'Negation Flag']) 

    df.to_excel(self.save_pred_out)


  def run(self):
    if self.type_neg == 'simple':
      self.my_own_negation()
    elif self.type_neg == 'professor':
      self.negation_professor_solution()

    self.save_pred()

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--type_neg', type=str, default='simple',
                        help='Choose mode of negation execution - simple or professor')
  parser.add_argument('--clean_type', type=str, default='no_cleaning',
                        help='Choose mode of data pre-processing - no_cleaning, lower_case, lower_stop, lower_steam,lower_stop_steam')

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  model = Sympton_detection(type_neg=args.type_neg.lower(), clean_type=args.clean_type.lower())
  model.run()



