#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:28:40 2018

@author: ethanshafron
"""

import geopandas as gpd
from ProcessingTweets import clean
import pandas as pd
import re
import collections
from operator import itemgetter
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
import spacy
from nltk.corpus import stopwords

def tokenize(corpus):
    """Separate list of tweet strings into lowercase words"""
    WordList = []
    for tweet in corpus:
        WordList.append(re.sub("[^\w]", " ",  tweet).split())
    FlattenedWordList = clean([item.lower() for sublist in WordList for item in sublist])
    return FlattenedWordList

def CountWordInCorpus(word, corpus):
    """Count frequency of a word in a corpus of tweet text"""
    counter = collections.Counter()
    for w in tokenize(corpus):
        counter[w] += 1
    return counter[word]

def get_intersection(oc, rc):
    """Get words common to both corpora/counts"""
    keys_x = set(oc.keys())
    keys_y = set(rc.keys())
    intersection = keys_x.intersection(keys_y)
    return intersection

def FigureOutPolarity(o, r, b):
    l = []
    for w in b:
        Numerator = o[w] - r[w]
        Dominator = o[w] + r[w]
        c = Numerator/Dominator
        l.append([w, c])
    return l

def lemmatize(l, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for tweet in l:
        doc = nlp(" ".join(tweet)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def remove_stops(l):
    tweets = [[word for word in simple_preprocess(str(tweet)) if word not in stop_words] for tweet in l]
    return tweets

def WordPolarity(word, o, r):
    """Calculates Yule coefficient given a word and two corpora """
    cw = (CountWordInCorpus(word, o) - CountWordInCorpus(word, r))/(CountWordInCorpus(word, o) + CountWordInCorpus(word, r))
    return cw

def TopYules(X,Y):
    XCounts = dict(collections.Counter(X).most_common())
    YCounts = dict(collections.Counter(Y).most_common())
    BothWordCounts = get_intersection(XCounts, YCounts)
    Polarity = FigureOutPolarity(XCounts, YCounts, BothWordCounts)
    
    YSlanted = sorted(Polarity, key=itemgetter(1))[0:10]
    XSlanted = sorted(Polarity, key=itemgetter(1), reverse = True)[0:10]
    
    PolarityXDF = pd.DataFrame(XSlanted, columns = ['X_Words', 'C_X'])
    PolarityYDF = pd.DataFrame(YSlanted, columns = ['Y_Words', 'C_Y'])
    PolarityBoth = pd.concat([PolarityXDF, PolarityYDF], axis=1, sort=False)
    print(PolarityBoth.to_string())
    return PolarityBoth
    
lemma = WordNetLemmatizer()

nlp = spacy.load('en', disable=['parser', 'ner'])
stop_words = stopwords.words('english')
stop_words.extend(['thankful', 'grateful','blessed', 'thanksgiving', 'amp', '', ' '])

TweetsDF = gpd.read_file('Data/CountyTweets/CountyTweets.shp', encoding = 'utf-8')
Transformed = TweetsDF.groupby('stateID')['text'].apply(list).reset_index()
Transformed = dict([(i,a) for i, a in zip(Transformed.stateID, Transformed.text)])

for state in Transformed:
    Transformed[state] = remove_stops(Transformed[state])
    Transformed[state] = lemmatize(Transformed[state])
    Transformed[state] = [tweet for state in Transformed[state] for tweet in state]
    Transformed[state] = clean(Transformed[state])

CaliCorpus = list(filter(None,Transformed['06']))
NYCorpus = list(filter(None,Transformed['36']))
FlorCorpus = list(filter(None,Transformed['12']))
TexCorpus = list(filter(None,Transformed['48']))
PennCorpus = list(filter(None,Transformed['42']))
VTCorpus = list(filter(None,Transformed['50']))

### Let's also make a corpus of all tweets
WholeCorpus = []
for state in Transformed:
    WholeCorpus.extend(Transformed[state])
WholeCorpus = list(filter(None, WholeCorpus))

CA_NY = TopYules(CaliCorpus, NYCorpus)
CA_FL = TopYules(CaliCorpus, FlorCorpus)
CA_TX = TopYules(CaliCorpus, TexCorpus)
CA_PA = TopYules(CaliCorpus, PennCorpus)
CA_VT = TopYules(CaliCorpus, VTCorpus) 

NY_FL = TopYules(NYCorpus, FlorCorpus)
NY_TX = TopYules(NYCorpus, TexCorpus)
NY_PA = TopYules(NYCorpus, PennCorpus)
NY_VT = TopYules(NYCorpus, VTCorpus)   

FL_TX = TopYules(FlorCorpus, TexCorpus)
FL_PA = TopYules(FlorCorpus, PennCorpus)
FL_VT = TopYules(FlorCorpus, VTCorpus)  

TX_PA = TopYules(TexCorpus, PennCorpus)
TX_PA = TopYules(TexCorpus, VTCorpus)

PA_VT = TopYules(PennCorpus, VTCorpus)

WH_CA = TopYules(WholeCorpus, CaliCorpus)
WH_NY = TopYules(WholeCorpus, NYCorpus)
WH_FL = TopYules(WholeCorpus, FlorCorpus)   
WH_TX = TopYules(WholeCorpus, TexCorpus) 
WH_PA = TopYules(WholeCorpus, PennCorpus)    
WH_VT = TopYules(WholeCorpus, VTCorpus)
    
    
    
    
    
    
    
    
    







