import codecs
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.externals import joblib
import lda

def readin_data(G, name_vector):
    return pd.read_csv("../../ds/new_groups/" + G, names=name_vector,sep="###",encoding="utf-8",engine='python')
    