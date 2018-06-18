from shutil import copyfile
import os
import numpy as np
from os import listdir
from os.path import isfile, join
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def load_filenames(file_name):
    f = open(file_name, 'r')
    filenames = []
    for line in f:
        filenames.append(line.replace('\n', ''))
    return filenames




src_dirc = 'origin-images-not-used-remaining' #this is where you images are
images_in_directory= [f for f in listdir(src_dirc) if isfile(join(src_dirc, f))]

image_dict = dict()
for image in images_in_directory:
	ind = image.find('_')
	if RepresentsInt(image[:ind]) and ind != -1: #check if the the first chunk is int or not
		print(image[:ind])
		if image[:ind] not in image_dict:
			image_dict[image[:ind]] = []
		image_dict[image[:ind]].append(image) #add the image that start with that number


image_list = load_filenames('poor.txt') #read image list
dest_dirc = 'poor-three' #the destination directory. You should creat it
print(image_list)
for image_name in image_list:
	if image_name in image_dict:
		print(image_name, image_dict[image_name])
		for image in image_dict[image_name]:
			copyfile(src_dirc + '/' + image, dest_dirc + '/' + image)



