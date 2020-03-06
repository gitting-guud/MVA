# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:37:02 2020

@author: Houcine's laptop
"""

###################################################### PREPROCESSING ROWS #########################################
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



def process_token(tok) :
  if len(re.findall("\d", tok))>0 :
    return "#"
  pro_tok = tok.lower()
  return pro_tok

def remove_func_tag(row):
  """remove functionnal tags from sentence"""
  func_tags_in_row = re.findall("\(\w+(-\w+)", row) # matches every tag that comes after a hyphen
  new_row = row
  for func_tag in func_tags_in_row :
    new_row = new_row.replace(func_tag, "")
  return new_row.lower()

def Binarisation_row(row) :
  tr = Tree.fromstring(remove_func_tag(row), remove_empty_top_bracketing=True)
  tr.chomsky_normal_form()
  tr.chomsky_normal_form(horzMarkov=2)
  tr.collapse_unary(collapsePOS=True, collapseRoot=True)
  row_rules = tr.productions()
  sentence_for_LM = tr.leaves()
  return row_rules, sentence_for_LM

def Binarisation_row_inference(row) :
  tr = Tree.fromstring(remove_func_tag(row), remove_empty_top_bracketing=True)
  tr.chomsky_normal_form()
  tr.chomsky_normal_form(horzMarkov=0)
  tr.collapse_unary(collapsePOS=True, collapseRoot=True)
  tree_to_compare = ' '.join(str(tr).split())
  
  new_tree_to_compare = tree_to_compare
  count_loops = 100
  c = 0
  while ("+" in new_tree_to_compare ) and (c<count_loops):
    c+=1
    func_tags_in_row = re.findall("\(\w+(\+\w+)", new_tree_to_compare)
    for func_tag in func_tags_in_row :
      new_tree_to_compare = new_tree_to_compare.replace(func_tag, "")

  func_tags_in_row = re.findall(r'\|(.*?)\>', new_tree_to_compare)
  new_tree_to_compare_ = new_tree_to_compare
  for func_tag in func_tags_in_row :
    new_tree_to_compare_ = new_tree_to_compare_.replace(func_tag, "")

  new_tree_to_compare_ = re.sub("(\|\>)","",new_tree_to_compare_)

  return new_tree_to_compare_ , [process_token(w) for w in tr.leaves()]

def normalize_counts(dico):
    """Converts counts to probabilities"""
    for key in dico.keys():
        denominator = sum(list(dico[key].values()))
        new_dicokey = {k:v/denominator for k, v in dico[key].items()}
        dico[key] = new_dicokey
    return dico

def train_language_model(sentences):
  sentences = [[s.lower() for s in sent] for sent in sentences]
  train_grams, vocab = padded_everygram_pipeline(2, sentences)
  model = lm.Laplace(2)
  model.fit(train_grams, vocab)
  return model
