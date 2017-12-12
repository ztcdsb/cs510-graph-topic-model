import pandas as pd
import codecs
import numpy as np
from sklearn import cross_validation as cv 
import createGraphFeatures as graph
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
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

# number of words 
sliding_window = 2

# data format
name_vector = ['id','class', 'text']

# read in data and convert them to above data format
train_ds=helper.readin_data(train_file, name_vector).as_matrix()
train_unsupervised=helper.readin_data(unsupervised_train_data, name_vector)

train_ds, eval_ds = cv.train_test_split(train_ds, train_size=0.8, test_size=0.2,random_state=42)
train=pd.DataFrame(train_ds, columns = name_vector)
eval_set=pd.DataFrame(eval_ds, columns = name_vector)

# labels from training data
y_train=train['class']
y_eval_set=eval_set['class']

# idf parameter we are using  
# "no"
# "idf"
# "icw"
idf_pars = [sys.argv[6]]
b=0.003
# ways to generate the topology graph 
centrality_pars = ["degree_centrality"]
for idf_par in idf_pars:
	for i in range(len(num_features)):	
		for centrality_par in centrality_pars:
			try:
				print("idf:"+idf_par)
				print("centrality_par:"+centrality_par)
			
				centrality_col_par = "eigenvector_centrality"
				print("centrality_col_par:"+centrality_col_par)

				# Get the number of documents based on the dataframe column size
				num_documents = train_unsupervised.shape[0]

				# Initialize an empty list to hold the clean-preprocessed documents
				clean_train_unsuper_documents = []
				unique_words = []
				print("Computing unique words")

				# Loop over each document; create an index i that goes from 0 to the length of the document list 
				clean_train_unsuper_documents=train_unsupervised['text'].tolist()
				
				# remove stop words and initialize vectorizer to convert non-numerical data into numerical data
				vectorizer = CountVectorizer(max_features=num_features[i],stop_words='english')
				vectorizer.fit_transform(train_unsupervised['text'])
				unique_words=vectorizer.get_feature_names()

				print("Unique words:"+str(len(unique_words)))

				# generate the topology graph
				features = graph.createGraphFeatures(num_documents,clean_train_unsuper_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par).astype(int)
			
				print("Train unsupervised gow computed")
			except Exception as e:
				print(e)
				print("train unsupervised error")
			try:
				print(train.shape)

				# Get the number of documents based on the dataframe column size
				num_train_documents = train.shape[0]

				# Initialize an empty list to hold the clean-preprocessed documents
				clean_train_documents = []
				clean_train_documents=train['text'].tolist()
			
				print("Computing train gow")
				train_features =  graph.createGraphFeatures(num_train_documents,clean_train_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par).astype(int)
				print("Train  gow computed")
			
				print(train_features.shape)
			except Exception as e:
				print(e)
				print("train gow error")
			try:
				print(eval_set.shape)

				# Get the number of documents based on the dataframe column size
				num_eval_documents = eval_set.shape[0]

				# Initialize an empty list to hold the clean-preprocessed documents
				clean_eval_documents = []
				clean_eval_documents=eval_set['text'].tolist()

				print("Computing eval_set gow")
				eval_features =  graph.createGraphFeatures(num_eval_documents,clean_eval_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par).astype(int)
				print(eval_features.shape)
				print("eval_set gow computed")

			except Exception as e:
				print(e)
				print("eval_set gow error")
			try:
				label = preprocessing.LabelEncoder()
				y_train= label.fit_transform(y_train)
				y_eval_set= label.transform(y_eval_set)
				
			except Exception as e:
				print(e)
				print("Preprocessing error")

			for z in range(len(iterations)):
		
					for j in range(len(num_topics)):
						print(num_features[i])
						print(num_topics[j])
						print(iterations[z])
						iters=iterations[z]
						n_topics = num_topics[j]
					try:
	
						print("convert text into sparse matrix...")

						model = lda.LDA(n_topics=num_topics[j], n_iter=iters, random_state=1)
						train_learn_lda=model.fit_transform(features)
		
						print("Saving the object")
						joblib.dump(model, 'objects_lda_learn_news/lda_learn_news_tw_'+str(centrality_par)+"_"+str(sliding_window)+"_"+str(iters)+'_'+str(num_features[i])+"_"+str(num_topics[j])+".pkl") 
						print('objects_lda_learn_news/lda_learn_news_tw_'+str(centrality_par)+"_"+str(sliding_window)+"_"+str(iters)+'_'+str(num_features[i])+"_"+str(num_topics[j])+".pkl")
						print("transforming train and eval_set")
					except Exception as e:
						print(e)
						print("Error while computing learn features")