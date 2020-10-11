"""
Week 8, in-class task
CS 584: Applied BioNLP
@author Abeed Sarker
email: abeed.sarker@dbmi.emory.edu

Metamap example based on pymetamap
Created: 10/1/2020
***DO NOT REDISTRIBUTE***

"""
from pymetamap import MetaMap
import pandas as pd

mm = MetaMap.get_instance('/Users/thiago/Documents/METAMAP/public_mm/binmetamap18')
sents = ['john had a heart attack and he has high blood pressure']
concepts,errors = mm.extract_concepts(sents)
for c in concepts:
    print (c.index,c.score,c.preferred_name,c.cui,c.semtypes)

print('***---***')
f_path = './piboso-train.csv'
df = pd.read_csv(f_path)
texts = df['Text']
ids = df['Document']
sents = df['Sentence']
from collections import defaultdict

concepts_per_sent = defaultdict(list)
for t,s,i in zip(texts,sents,ids):
    concepts,errors = mm.extract_concepts([t])
    print(concepts)
    print(t)
    for c in concepts:

        print(c.index, c.score, c.preferred_name, c.cui, c.semtypes,errors)
    print('--------')
