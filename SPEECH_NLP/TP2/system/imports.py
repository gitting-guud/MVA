# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:38:58 2020

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