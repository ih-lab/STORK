import cv2
import glob
import numpy as np



i = 0
for image in glob.glob('/Users/pek2012/Desktop/poor/*.jpg'):
	img = cv2.imread(image)
	name = image.split('/')[-1].split('.')[0]

	filename = '/Users/pek2012/Desktop/poor/poor_%s.jpg' % name
	
	cv2.imwrite(filename, img)
	
i += 1
print 'done'