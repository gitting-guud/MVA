# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:37:26 2020

@author: Houcine's laptop
"""

################################### CYK ###################################################

import utils
import oov

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

def probabilistic_CYK(tokens, TERMINAL, NONTERMINAL, flipped_NONTERMINAL,
                      oov_param_vocabulary, oov_param_k_leven, oov_param_words_polyglot,
                      oov_param_nearest_neighbors_fit, oov_param_polyglot_emb, oov_param_language_model
                      ):
    P = {}    
    back = {}
    all_possible_parents = list(set(list(TERMINAL.keys())+list(NONTERMINAL.keys())))
    for A in all_possible_parents :
        P[A] = np.zeros((len(tokens)+1,len(tokens)+1))

    for j in range(1, len(tokens)+1):
        if j == 1 :
          prev_word = ""
        else :
          prev_word = utils.process_token(tokens[j-2])
        current_word = oov.word_to_process(tokens[j-1],
                                        vocabulary=oov_param_vocabulary,
                                        k_leven=oov_param_k_leven,
                                        words_polyglot=oov_param_words_polyglot,
                                        nearest_neighbors_fit=oov_param_nearest_neighbors_fit,
                                        polyglot_emb = oov_param_polyglot_emb,
                                        language_model=oov_param_language_model,
                                        previous_word=prev_word)

        for a, A in enumerate(TERMINAL.keys()):
            children_A = TERMINAL[A].keys()
            if current_word in children_A :
              P[A][j-1,j] = TERMINAL[A][current_word]

            added = True
            while added:
                added = False
                for b,B in enumerate(NONTERMINAL.keys()):
                    if (P[B][j-1,j] > 0) and (B in children_A) :
                        PROB = NONTERMINAL[B][A] * P[B][j-1,j]
                        if PROB > P[A][j-1,j]:
                            P[A][j-1,j] = PROB
                            back[(A,j-1,j)] = b
                            added = True

        for i in reversed(range(0,j-1)):
            for k in range(i+1,j):
              Bs = np.array(all_possible_parents)[np.array(list(P.values()))[:,i,k] > 0]
              Cs = np.array(all_possible_parents)[np.array(list(P.values()))[:,k,j] > 0]
              for B in Bs:
                for C in Cs: 
                  if (B + " " + C) in flipped_NONTERMINAL.keys() :
                    possible_parents = list(flipped_NONTERMINAL[B + " " + C].keys())
                  else :
                    continue

                  for parent in possible_parents :
                    proba_A_BC = NONTERMINAL[parent][B + " "+ C]
                    proba_B = P[B][i,k]
                    proba_C = P[C][k,j]

                    if (P[parent][i,j] < proba_A_BC * proba_B * proba_C):
                        P[parent][i,j] = proba_A_BC * proba_B * proba_C
                        back[(parent,i,j)] = [k,B,C]
    return P,back

def MOST_Probable_Tree(tokens, back, START_TAG, start, end, S=''):
    
    keys = np.array(list(back.keys()))

    if (START_TAG, start, end) in back :
      L = back[(START_TAG, start, end)]
      k, TAG_1, TAG_2 = L[0], L[1], L[2]
      S += '(' + TAG_1 + " " + MOST_Probable_Tree(tokens, back, TAG_1, start, k) + ") "
      S += '(' + TAG_2 + " " + MOST_Probable_Tree(tokens, back, TAG_2, k, end) + ")"
      return S
    else :
      return tokens[start]

def inference_for_comparison(STRING) :
  return '(sent '+STRING+')'

