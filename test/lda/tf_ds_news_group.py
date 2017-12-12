import codecs
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv 
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import lda
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import Imputer
from sklearn import svm
import sys
import helper

def avgLen(n, m, matrix_result):
    re_length = 0
    for i in range(n):
        re_length += matrix_result[i,:]

    re_length = re_length/n
    return re_length

def doc_len_norm_implementation(n, m, matrix_result):
    b = 0.03
    avg_length = avgLen(n, m, matrix_result)
    
    for i in range(n):
        doc_length_temp = np.sum(matrix_result[i,:])
        doc_norm_temp = 1.0/(1 - b + (b* (doc_length_temp/avg_length)))
        matrix_result[i,:] = matrix_result[i,:] * doc_norm_temp
    return matrix_result



def idf_implementation(n, m, matrix_result):
    for i in range(m):
        num_temp = np.count_nonzero(matirx_result[:,i])
        idf_temp = np.log10((n + 1)/num_temp)
        doc_length = np.sum(matirx_result[:,i])
        matrix_result[:,i] = matrix_result[:,i] * idf_temp
        
    matrix_result = doc_len_norm_implementation(matrix_result)
    return matrix_reuslt

def euclidean_similarity(v1, v2):
    ret_val = la.norm(v1, v2)
    return ret_val


def cos_similarity(v1, v2):
    ret_val = 1.0 - np.dot(v1, v2)/(la.norm(v1)*la.norm(v2))
    return ret_val

def tf_idf_similarity(test_unigrams_lda):
    matrix_temp = test_unigrams_lda.toarray()
    n, m = np.shape(matrix_temp)
    matrix_result = np.zeros((n,n)) 
    
    for i in range(n):
        for j in range(i):
            temp = cos_similarity(matrix_temp[i,:], matrix_temp[j,:])
            matrix_result[i,j] = temp
            matrix_result[j,i] = temp

    matrix_result = idf_implementation(n,m,matrix_result)
    return matrix_result

def doc_euclid_similarity(test_unigrams_lda):
    matrix_temp = test_unigrams_lda.toarray()
    n, m = np.shape(matrix_temp)
    matrix_result = np.zeros((n,n))

    for i in range(n):
        for j in range(i):
            temp = euclidean_similarity(matrix_temp[i,:], matrix_temp[j,:])
            matrix_result[i,j] = temp
            matrix_result[j,i] = temp
    return matrix_result

def doc_cos_similarity(test_unigrams_lda):
    matrix_temp = test_unigrams_lda.toarray()
    n, m = np.shape(matrix_temp)
    matrix_result = np.zeros((n,n))

    for i in range(n):
        for j in range(i):
            temp = cos_similarity(matrix_temp[i,:], matrix_temp[j,:])
            matrix_result[i,j] = temp
            matrix_result[j,j] = temp
            print(matrix_result[i,j])

    return matrix_result


def lda_smooth(theta,doc_sim):

    res_theta = copy.deepcopy(theta)
    num_doc = len(doc_sim)
    num_topic = len(theta[0])
    print('num_doc, num_topic')
    print(num_doc, num_topic)
    
    for i in range(num_doc):
        max_sim = np.argmin(doc_sim[i])
        for n in range(num_topic):
                if i != max_sim:
                    res_theta[i][n] += (2 - doc_sim[i][max_sim])/2 * theta[max_sim][n] 
        
    print(res_theta)
    print(theta)
    return res_theta

def printTofile(dataset,classifier,classifierParams,nfolds,vectorizerType,vectorizerParams,topicsType,topicsNumber,topicsParams,accuracy):

	with codecs.open("tf_standard_seed_accuracy_newsgroup_best.txt", "a", "utf-8") as myfile:
		myfile.write(dataset+"\t"+classifier+"\t"+str(classifierParams)+"\t"+str(nfolds)+"\t"+str(vectorizerType)+"\t"+str(vectorizerParams)+"\t"+topicsType+"\t"+str(topicsNumber)+"\t"+str(topicsParams)+"\t"+str(accuracy)+"\n")


def printAccuracies(data_train, data_test,lab_col,dataset):
	
	train_labels=data_train[:,lab_col]
	train_reviews=data_train[:,2]
	test_labels=data_test[:,lab_col]
	test_reviews=data_test[:,2]
	label = preprocessing.LabelEncoder()
	train_labels= label.fit_transform(train_labels)
	test_labels= label.transform(test_labels)
	
	#lda
	unsupervised_train_data = sys.argv[1] #"news_groups_unsupervised.txt"
	train_data_unsupervised=helper.readin_data(unsupervised_train_data, name_vector).as_matrix()


	list_topics=[int(sys.argv[2])] #100
	list_features=[int(sys.argv[3])] #5000
	iterations=[int(sys.argv[4])] #1000

	for i in range(len(iterations)):
		for j in range(len(list_features)):
			for z in range(len(list_topics)):
				try:
					iters=iterations[i]
					max_features_nr=list_features[j]
					topics_nr=list_topics[z]

					print("load the object")
					model=joblib.load('../../learn/lda/objects_lda_learn_news/lda_learn_news_tf_'+str(iters)+'_'+str(max_features_nr)+'_'+str(topics_nr)+'.pkl') 
					print("object loaded")
					vectorizer = CountVectorizer(max_features=max_features_nr,stop_words='english')
					train_learn_unigrams_lda=vectorizer.fit_transform(train_data_unsupervised[:,2])
					train_unigrams_lda=vectorizer.transform(train_reviews)
					test_unigrams_lda=vectorizer.transform(test_reviews)

					similary_matrix = doc_cos_similarity(test_unigrams_lda[0:30])

					print("vectorized")

					train_lda=model.transform(train_unigrams_lda.toarray()) # model.fit_transform(X) is also available

					test_lda=model.transform(test_unigrams_lda.toarray()) # model.fit_transform(X) is also available
					print("transformed")

					vocab = vectorizer.get_feature_names()
					topic_word = model.topic_word_  # model.components_ also works
					

					imp=Imputer(missing_values='NaN', strategy='mean') 
					X2 = imp.fit_transform(train_lda)
					X3 = imp.transform(test_lda)
					
					X3 = lda_smooth(X3,similary_matrix)

					clf =svm.SVC(kernel="linear")
					
					clf.fit(X2, np.array(train_labels).astype(int))     
					ypred = clf.predict(X3)

					print(accuracy_score(test_labels.astype(int), ypred))

					acc=accuracy_score(test_labels.astype(int), ypred)
					macro=precision_recall_fscore_support(test_labels.astype(int), ypred, average='macro')
					micro=precision_recall_fscore_support(test_labels.astype(int), ypred, average='micro')
					macrof1=f1_score(test_labels.astype(int), ypred, average='macro')  
					microf1=f1_score(test_labels.astype(int), ypred, average='micro')  
					weightedf1=f1_score(test_labels.astype(int), ypred, average='weighted')
					nonef1=f1_score(test_labels.astype(int), ypred, average=None)	


					with codecs.open("tf_lda_standard_"+str(len(vectorizer.get_feature_names()))+"_"+str(topics_nr)+"_"+str(iters)+"_"+dataset+".txt", "a", "utf-8") as myfile:
						myfile.write(dataset+"\t"+"svm.SVC()"+"\t"+"linear"+"\t"+"TF"+"\t"+str(len(vectorizer.get_feature_names()))+"\t"+"LDA"+"\t"+str(topics_nr)+"\t"+str(iters)+ '\t' +str(acc)+'\t'+str(macro[0])+'\t'+str(macro[1])+'\t'+str(micro[0])+'\t'+str(micro[1])+'\t'+str(macrof1)+'\t'+str(microf1)+'\t'+str(weightedf1)+'\t'+str(nonef1[0])+'\t'+str(nonef1[1])+'\t' +"\n")
		 

				except Exception as e:
					print(e)
					print("Exception"+"\t"+dataset+ "LDA"+"\t"+str(topics_nr)+"\t"+str(iters)+"\t"+str(max_features_nr))
					raise

name_vector=['id','type','review']

train_ds=pd.read_csv("../../ds/new_groups/sdyg_news_groups_learn_first.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
test_ds=pd.read_csv("../../ds/new_groups/sdyg_news_groups_test_first.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
printAccuracies(train_ds,test_ds,1,"news_groups")

train_ds=pd.read_csv("../../ds/R8/r8_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
test_ds=pd.read_csv("../../ds/R8/r8_test.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
printAccuracies(train_ds,test_ds,1,"R8")

train_ds=pd.read_csv("../../ds/R52/r52_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
test_ds=pd.read_csv("../../ds/R52/r52_test.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
printAccuracies(train_ds,test_ds,1,"R52")


train_ds=pd.read_csv("../../ds/bbc_news/bbc_news.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
train_ds, test_ds = cv.train_test_split(train_ds, train_size=0.8, test_size=0.2,random_state=42)
printAccuracies(train_ds,test_ds,1,"bbc_news")


