import pandas as pd
import random

cols = ['id','class', 'text']
train_ds=pd.read_csv("../../ds/new_groups/news_groups_all.txt",names=cols,sep="###",encoding="utf-8",engine='python').as_matrix()
train=pd.DataFrame(train_ds, columns = cols)
count = 0
topic = []

for index, row in train.iterrows():
	if row[1] not in topic:
		topic.append(row[1])
selected_topic = []
'''
['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
omp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 
'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 
'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 
'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 
'talk.religion.misc']
'''
selected_topic.append(topic[2])
selected_topic.append(topic[7])
selected_topic.append(topic[9])
selected_topic.append(topic[13])
selected_topic.append(topic[17])
'''
f = open('news_groups_new.txt','a')
for index, row in train.iterrows():
	f.write(str(count))
	f.write('###' + row[1] +'###' + row[2] + '\n')
	count+=1
'''
un_train_ds=pd.read_csv("news_groups_new_few_second.txt",names=cols,sep="###",encoding="utf-8",engine='python').as_matrix()
un_train=pd.DataFrame(un_train_ds, columns = cols)
f = open('news_groups_new_selected_unsupervised_second.txt','a')
for index, row in un_train.iterrows():
	if(row[1] in selected_topic):
		f.write(str(count))
		f.write('###' + 'unsupervised' +'###' + row[2] + '\n')
		count+=1

'''
f = open('news_groups_new_few_second.txt','a')
for index, row in train.iterrows():
	if(row[1] in selected_topic):
		f.write(str(count))
		f.write('###' + row[1] +'###' + row[2] + '\n')
		count+=1
'''

topic1 = 0
topic6 = 0
topic8 = 0
topic12 = 0
topic16 = 0
'''
f = pd.read_csv("news_groups_new_few_second.txt",names=cols,sep="###",encoding="utf-8",engine='python').as_matrix()
af=pd.DataFrame(f, columns = cols)


for index, row in train.iterrows():
	if(row[1]==selected_topic[0]):
		topic1+=1
	if(row[1]==selected_topic[1]):
		topic6+=1
	if(row[1]==selected_topic[2]):
		topic8+=1
	if(row[1]==selected_topic[3]):
		topic12+=1
	if(row[1]==selected_topic[4]):
		topic16+=1


sub_topic1 = 0
sub_topic6 = 0
sub_topic8 = 0
sub_topic12 = 0
sub_topic16 = 0
target_learn= open('sdyg_news_groups_learn_second.txt', 'w')
target_test =open('sdyg_news_groups_test_second.txt', 'w')
for i in range(1):
	for j in range(1):
		for index, row in af.iterrows():
			if(row[1] in selected_topic):
				if(row[1]==selected_topic[0] and sub_topic1 < (int) (topic1*0.6)):
					target_learn.write(str(count))
					target_learn.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1
					sub_topic1+=1
				if(row[1]==selected_topic[0] and sub_topic1 >= (int) (topic1*0.6)):
					target_test.write(str(count))
					target_test.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1

				if(row[1]==selected_topic[1] and sub_topic6 < (int) (topic6*0.6)):
					target_learn.write(str(count))
					target_learn.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1
					sub_topic6+=1
				if(row[1]==selected_topic[1] and sub_topic6 >= (int) (topic6*0.6)):
					target_test.write(str(count))
					target_test.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1

				if(row[1]==selected_topic[2] and sub_topic8 < (int) (topic8*0.6)):
					target_learn.write(str(count))
					target_learn.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1
					sub_topic8+=1
				if(row[1]==selected_topic[2] and sub_topic8 >= (int) (topic8*0.6)):
					target_test.write(str(count))
					target_test.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1

				if(row[1]==selected_topic[3] and sub_topic12 < (int) (topic12*0.6)):
					target_learn.write(str(count))
					target_learn.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1
					sub_topic12+=1
				if(row[1]==selected_topic[3] and sub_topic12 >= (int) (topic12*0.6)):
					target_test.write(str(count))
					target_test.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1

				if(row[1]==selected_topic[4] and sub_topic16 < (int) (topic16*0.6)):
					target_learn.write(str(count))
					target_learn.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1
					sub_topic16+=1
				if(row[1]==selected_topic[4] and sub_topic16 >= (int) (topic16*0.6)):
					target_test.write(str(count))
					target_test.write('###' + row[1] +'###' + row[2] + '\n')
					count+=1
'''


'''

with open('news_groups_new_few.txt', 'r') as source:
	data = [(random.random(), line) for line in source]
data.sort()
with open('womenbuyiyang_news_groups_new_few.txt', 'w') as target:
	for _, line in data:
		target.write(line)
'''

'''
f = open('news_groups_new_few.txt','a')
for index, row in train.iterrows():
	if(row[1] in selected_topic):
		f.write(str(count))
		f.write('###' + row[1] +'###' + row[2] + '\n')
		count+=1
'''

