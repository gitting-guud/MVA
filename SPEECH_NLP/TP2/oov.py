# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:36:59 2020

@author: Houcine's laptop
"""

#########################OOV #################
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


def Levenshtein(str1,str2):
  T = np.zeros((len(str1) +1,len(str2) +1))
  for i in range(len(str1) +1):
      T[i,0] = i
  
  for j in range(len(str2) +1):
      T[0,j] = j
  
  for i in range(1,len(str1) +1):
      for j in range(1,len(str2) +1):
          if str1[i-1]==str2[j-1]:
              T[i,j]= min(T[i-1,j]+1,T[i,j-1]+1,T[i-1,j-1])
          else:
              T[i,j] =min(T[i-1,j]+1,T[i,j-1]+1,T[i-1,j-1]+1)
  return T[-1,-1]

def is_present_in_vocabulary(word, vocabulary) :
  if word in vocabulary :
    return True
  else:
    return False

def k_distant_in_vocab_levenstein(word, vocabulary, k=3) :
  distances = [Levenshtein(word, str(v)) for i, v in enumerate(vocabulary)]
  indices = np.array(distances) <= k
  if (indices == False).all():
    return None
  return np.array(vocabulary)[indices]

def word_to_process(input_word,
                    vocabulary,
                    k_leven,
                    words_polyglot,
                    nearest_neighbors_fit,
                    polyglot_emb,
                    language_model,
                    previous_word="") :

  if is_present_in_vocabulary(input_word, vocabulary):
    return input_word
  else :
    closest_levenstein = k_distant_in_vocab_levenstein(input_word, vocabulary, k_leven)
    if closest_levenstein is not None :
      if previous_word == "" :
        return np.random.choice(closest_levenstein)
      else :
        lm_scores = [language_model.score(w, context=[previous_word]) for w in closest_levenstein]
        return closest_levenstein[np.argmax(lm_scores)]

    else :
      if input_word in words_polyglot :
        id_input_word_in_polyglot = words_polyglot.index(input_word)
        _, indices_in_polyglot = nearest_neighbors_fit.kneighbors([polyglot_emb[id_input_word_in_polyglot]])
        for i in indices_in_polyglot[0] :
          if i > len(words_polyglot) :
            continue
          if words_polyglot[i] in vocabulary :
            return words_polyglot[i]
        return None
      else :
        return None
