import codecs
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.externals import joblib
import lda
import helper
import argparse
import sys

# input file name
train_file = sys.argv[1] #"news_groups_train.txt"
unsupervised_train_data = sys.argv[2] #"news_groups_unsupervised.txt"

#number of topic
num_topics=[int(sys.argv[3])] #100

# number of stop words need to remove
num_features=[int(sys.argv[4])] #5000

# number of iteration to train the model
iterations=[int(sys.argv[5])] #1000


# data format
name_vector=['id','type','review']

# read in data and convert them to above data format
train_ds=helper.readin_data(train_file, name_vector).as_matrix()
train_data_unsupervised=helper.readin_data(unsupervised_train_data, name_vector).as_matrix()



train_ds, eval_ds = cv.train_test_split(train_ds, train_size=0.8, test_size=0.2,random_state=42)

# convert type of training data into numerical format
label = preprocessing.LabelEncoder()
train_ds[:,0]= label.fit_transform(train_ds[:,1])
eval_ds[:,0]= label.transform(eval_ds[:,1])

for i in range(len(num_features)):
    for j in range(len(num_topics)):
        for z in range(len(iterations)): 
            iters=iterations[z]
            print(num_features[i])
            print(num_topics[j])
    
    
        # remove stop words and initialize vectorizer to convert non-numerical data into numerical data
        vectorizer = CountVectorizer(max_features=num_features[i],stop_words='english')
        
        #convert reviews of all data into numerical format 
        train_learn_unigrams=vectorizer.fit_transform(train_data_unsupervised[:,2])
        train_unigrams=vectorizer.transform(train_ds[:,2])
        eval_unigrams=vectorizer.transform(eval_ds[:,2])

        print("start lda")
        model = lda.LDA(n_topics=num_topics[j], n_iter=iters, random_state=1)
        train_learn_lda=model.fit_transform(train_learn_unigrams) # model.fit_transform(X) is also available
        print("saving the object")
        joblib.dump(model, 'objects_lda_learn_news/lda_learn_news_tf_'+str(iters)+'_'+str(num_features[i])+"_"+str(num_topics[j])+".pkl") 
        