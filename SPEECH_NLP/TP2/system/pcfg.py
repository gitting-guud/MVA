# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:41:41 2020

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

import utils


def distinguish_terminal_non_terminal(total_rules) :

  terminal_rules = {}
  non_terminal_rules = {}
  for rule in total_rules :
    if rule.is_lexical() :
      token_processed = utils.process_token(rule.rhs()[0])
      if str(rule.lhs()) not in terminal_rules.keys() :
        terminal_rules[str(rule.lhs())] = {token_processed : 1}
      else :
        if token_processed not in terminal_rules[str(rule.lhs())].keys() :
          terminal_rules[str(rule.lhs())][token_processed] = 1
        else :
          terminal_rules[str(rule.lhs())][token_processed] += 1
    else :
      ### There are no unary rules as we collapsed them
      B, C = rule.rhs()
      BC= str(B) + " " + str(C)
      if str(rule.lhs()) not in non_terminal_rules.keys() :
        non_terminal_rules[str(rule.lhs())] = {BC: 1}
      else :
        if B not in non_terminal_rules[str(rule.lhs())].keys() :
          non_terminal_rules[str(rule.lhs())][BC]= 1
        else :
          non_terminal_rules[str(rule.lhs())][BC] += 1

  return non_terminal_rules, terminal_rules

def distinguish_terminal_non_terminal_remove_bars_plus(total_rules) :

  terminal_rules = {}
  non_terminal_rules = {}
  for rule in total_rules :
    parent = re.sub(r'(\|).*$', "", str(rule.lhs()))
    while "+" in parent :
      parent = re.sub(r'(\+).*$', "", parent)
    if rule.is_lexical() :
      token_processed = utils.process_token(rule.rhs()[0])
      if parent not in terminal_rules.keys() :
        terminal_rules[parent] = {token_processed : 1}
      else :
        if token_processed not in terminal_rules[parent].keys() :
          terminal_rules[parent][token_processed] = 1
        else :
          terminal_rules[parent][token_processed] += 1
    else :
      ### There are no unary rules as we collapsed them
      # print(rule.rhs())
      B, C = rule.rhs()
      B = re.sub(r'(\|).*$', "", str(B))
      C = re.sub(r'(\|).*$', "", str(C))

      while "+" in B :
        B = re.sub(r'(\+).*$', "", str(B))
      while "+" in C :
        C = re.sub(r'(\+).*$', "", str(C))

      BC= str(B) + " " + str(C)
      if parent not in non_terminal_rules.keys() :
        non_terminal_rules[parent] = {BC: 1}
      else :
        if B not in non_terminal_rules[parent].keys() :
          non_terminal_rules[parent][BC]= 1
        else :
          non_terminal_rules[parent][BC] += 1

  return non_terminal_rules, terminal_rules


def grammar_rules_lexicon_rules_probabilities(corpus) :

  total_rules_ = []
  sentences_for_LM = []
  for sentence in corpus :
    rules_sentence, sentence_for_LM = utils.Binarisation_row(sentence)
    total_rules_.extend(rules_sentence)
    sentences_for_LM.append(sentence_for_LM)
  # non_terminal_rules, terminal_rules = distinguish_terminal_non_terminal(total_rules_)
  non_terminal_rules, terminal_rules = distinguish_terminal_non_terminal_remove_bars_plus(total_rules_)
  return utils.normalize_counts(non_terminal_rules), utils.normalize_counts(terminal_rules), sentences_for_LM