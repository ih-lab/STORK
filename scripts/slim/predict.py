import tensorflow as tf
slim = tf.contrib.slim
import sys
import os
#import matplotlib.pyplot as plt
import numpy as np
import os
from nets import inception
from preprocessing import inception_preprocessing
from os import listdir
from os.path import isfile, join
from os import walk
#os.environ['CUDA_VISIBLE_DEVICES'] = '' #Uncomment this line to run prediction on CPU.
session = tf.Session()

def get_test_images(mypath):
	return [mypath + '/' + f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find('.jpg') != -1]

def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f,'rb').read(), channels=3)
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return session.run([out])[0]


if __name__ == '__main__':
	

	if len(sys.argv) != 6:
		print("The script needs five arguments.")
		print("The first argument should be the CNN architecture: v1, v3 or inception_resnet2")
		print("The second argument should be the directory of trained model.")
		print("The third argument should be directory of test images.")
		print("The  fourth argument should be output file for predictions.")
		print("The  fifth argument should be number of classes.")
		exit()
	deep_lerning_architecture = sys.argv[1]
	train_dir = sys.argv[2]
	test_path = sys.argv[3]
	output = sys.argv[4]
	nb_classes = int(sys.argv[5])

	if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
		image_size = 224
	else:
		if deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3" or deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
			image_size = 299
		else:
			print("The selected architecture is not correct.")
			exit()


	print('Start to read images!')
	image_list = get_test_images(test_path)
	processed_images = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

	if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
		with slim.arg_scope(inception.inception_v1_arg_scope()):
			logits, _ = inception.inception_v1(processed_images, num_classes=nb_classes, is_training=False)

	else:
		if deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3":
			with slim.arg_scope(inception.inception_v3_arg_scope()):
				logits, _ = inception.inception_v3(processed_images, num_classes=nb_classes, is_training=False)
		else:
			if deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
				with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
					logits, _ = inception.inception_resnet_v2(processed_images, num_classes=nb_classes, is_training=False)

	def predict_fn(images):
	    return session.run(probabilities, feed_dict={processed_images: images})

	probabilities = tf.nn.softmax(logits)
	checkpoint_path = tf.train.latest_checkpoint(train_dir)
	init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,slim.get_variables_to_restore())
	init_fn(session)
	print('Start to transform images!')
	images = transform_img_fn(image_list)

	fto = open(output, 'w')
	print('Start doing predictions!')
	preds = predict_fn(images)
	print (len(preds))
	for p in range(len(preds)):
		print (image_list[p], preds[p,:], np.argmax(preds[p,:]))
		fto.write(image_list[p])
		for j in range(len(preds[p,:])):
			fto.write('\t' + str(preds[p, j]))
		fto.write('\n')

	fto.close()