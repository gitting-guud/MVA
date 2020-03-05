# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:37:49 2020

@author: Houcine's laptop
"""
import re
import pickle
import nltk 
import sys

import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.model_selection import train_test_split


from nltk import Tree, lm
from nltk.lm.preprocessing import padded_everygram_pipeline

import oov
import cyk

import utils 
import pcfg


PARAMETERS = {"seed" : 40 ,
              "NN" : 20,
              "levenstein_distance": 4}


sentences = list(pd.read_table('./sequoia-corpus+fct.mrg_strict', header=None)[0].values)

train, dev_test = train_test_split(sentences, train_size=.8, random_state=PARAMETERS["seed"])
dev, test = train_test_split(dev_test, train_size=.5, random_state=PARAMETERS["seed"])


words_polyglot, embeddings = pickle.load(open('polyglot-fr.pkl', 'rb'), encoding='latin1')
words_polyglot_lower = {v.lower() for v in words_polyglot}

from sklearn.neighbors import NearestNeighbors


normalised_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
NN = PARAMETERS["NN"]
knn_clf=NearestNeighbors(n_neighbors=NN)
knn_clf.fit(normalised_emb)


non_terminal_rules, terminal_rules, sentences_for_LM = pcfg.grammar_rules_lexicon_rules_probabilities(train+dev)

flipped_terminal_rules = defaultdict(dict)
for key, val in terminal_rules.items():
    for subkey, subval in val.items():
        flipped_terminal_rules[subkey][key] = subval

flipped_non_terminal_rules = defaultdict(dict)
for key, val in non_terminal_rules.items():
    for subkey, subval in val.items():
        flipped_non_terminal_rules[subkey][key] = subval
        
        
  
LM = utils.train_language_model(sentences_for_LM)
#%%
inferences = []
truths = []
corpus = test
for i in range(len(corpus)) :
    sys.stdout.write('\r'+str(i)+"/"+str(len(corpus)))
    truth, tokens = utils.Binarisation_row_inference(corpus[i])
    P,back = cyk.probabilistic_CYK(tokens, 
                          TERMINAL=terminal_rules,
                          NONTERMINAL=non_terminal_rules,
                          flipped_NONTERMINAL=flipped_non_terminal_rules,
                          oov_param_vocabulary=list(flipped_terminal_rules.keys()),
                          oov_param_k_leven=PARAMETERS["levenstein_distance"],
                          oov_param_words_polyglot=list(words_polyglot_lower),
                          oov_param_nearest_neighbors_fit=knn_clf,
                          oov_param_polyglot_emb = normalised_emb,
                          oov_param_language_model = LM)
    infered = cyk.inference_for_comparison( cyk.MOST_Probable_Tree(tokens, back, "sent", 0, len(tokens), S=''))
    inferences.append(infered)
    truths.append(truth)


#%%
    
np.save( "evaluation_data.parser_output",inferences)