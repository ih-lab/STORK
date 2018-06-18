import numpy as np
import sys
from collections import defaultdict


def read_prediction(file_name, lables):

	wrong_counts = np.zeros((len(lables), len(lables)))
	label_int = dict()
	for i in range(len(lables)):
		label_int[lables[i]] = i
	lable_dict = dict()
	for i in range(len(lables)):
		lable_dict[i] = lables[i]
	evaluation = defaultdict(list)
	embryo_predict = dict() #for majority
	embryo_predict_max = dict()
	embryo_predict_nb = dict()
	embryo_cnt = dict()
	embryo_label = dict()
	f = open(file_name, 'r')
	for line in f:
		parsed = line.replace('\n', '').split('\t')
		score = []
		for i in range(1, len(parsed)):
			score.append(float(parsed[i]))
		image_name = parsed[0]
		embryo_name = image_name
		embryo_name = embryo_name.replace('../../Images/test/','') #embryo_name is the name of embryo. all the images are grouped as a single embryo_name.
		
		ind = embryo_name.rfind('_')
		embryo_name = embryo_name[:ind]
		ind = embryo_name.rfind('_')
		embryo_name = embryo_name[:ind]
		ind = embryo_name.find('_')
		embryo_name = embryo_name[ind+1:]

		if embryo_name not in embryo_predict:
			embryo_predict[embryo_name] = 0
			embryo_cnt[embryo_name] = 0
			embryo_predict_max[embryo_name] = [1,1,1,1] #for multiplication
			#embryo_predict_max[embryo_name] = [0,0,0,0] #for sum
			embryo_predict_nb[embryo_name] = [0,0,0,0] #for counting number of predicted 

		current_cnt = embryo_predict_nb[embryo_name]
		current_cnt[np.argmax(score)]+=1
		embryo_predict_nb[embryo_name] = current_cnt

		current_score = embryo_predict_max[embryo_name]
		for i in range(len(lables)):
			current_score[i] *= score[i] # multiplication
			#current_score[i] += score[i] # sum
		embryo_predict_max[embryo_name] = current_score

		t_label = -1
		image_name_ind = image_name.rfind('/')
		image_name = image_name[image_name_ind+1:]
		t_flag = True

		for lable in lables:
			if image_name.find(lable + '_') != -1 and t_flag:
				t_label = lable
				t_flag = False
		embryo_label[embryo_name] = t_label
		if t_label == -1:
			print (image_name)
			print('Error')
			exit()
		p_label = lable_dict[np.argmax(score)]
		embryo_cnt[embryo_name]+=1
		if p_label == t_label:
			embryo_predict[embryo_name]+=1
		#print(embryo_name, t_label, p_label, score, image_name)
		#if image_name in evaluation:
			#print(image_name)
		evaluation[image_name] = [t_label, score]
	c = 0
	t = 0
	for embryo_name in embryo_predict.keys():
		t += 1
		#if embryo_predict[embryo_name] > embryo_cnt[embryo_name] / 2: #check if more than half of them are from the correct lable
		if embryo_label[embryo_name] == lable_dict[np.argmax(embryo_predict_nb[embryo_name])]:
			c+=1
		else:
			#print(embryo_predict[embryo_name], embryo_name,embryo_label[embryo_name], lable_dict[np.argmax(embryo_predict_max[embryo_name])], embryo_predict_max[embryo_name], np.sum(embryo_predict_max[embryo_name]), embryo_predict_nb[embryo_name])
			wrong_counts[label_int[embryo_label[embryo_name]], np.argmax(embryo_predict_nb[embryo_name])] += 1
		#print(embryo_cnt[embryo_name])
	print("embryo accuracy:", c, t, c/ t)
	#print(len(evaluation))
	print("number of misclassified images from different classes:: ")
	print(wrong_counts)
	#print(np.sum(wrong_counts))
	return evaluation, lables, lable_dict


def acc(evaluation, lables, lable_dict):
	c = 0
	t = 0
	lables_count = dict()
	for label in lables:
		lables_count[label] = 0
	for image in evaluation:
		t_label = evaluation[image][0]
		p_label = lable_dict[np.argmax(evaluation[image][1])]
		if t_label == p_label:
			c+=1
		#print(image, t_label, p_label, c, t)
			lables_count[t_label]+=1
			#print(image, t_label, p_label, lables_count[t_label], evaluation[image][1])
		t+=1
		
	acc = 0
	if t != 0:
		acc = c * 1. / t
	#print(lables_count)
	return c, t, acc


if __name__ == '__main__':
	file_name = 'output.txt'
	
	lables1 = ['good','poor']
	evaluation1, lables1, lable_dict1 = read_prediction(file_name, lables1)
	print("accuracy per image: ", acc(evaluation1, lables1, lable_dict1))




