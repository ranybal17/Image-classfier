import cv2
import numpy as np

from tensorflow.keras.utils import to_categorical

def preprocess(filenames):
	data = []
	labels = []
	for f in filenames:
		img = cv2.imread(f)
		img_resize = cv2.resize(img, (128, 128))
		if f.split('/')[2][:3] == 'dog':
			labels.append(1)
		elif f.split('/')[2][:3] == 'cat':
			labels.append(0)
		data.append(img_resize)

	labels = to_categorical(labels)
	randy = np.random.permutation(len(data))
	labels = labels[randy, :]
	data = np.array(data)[randy, :, :, :]
	data = data / 255.0

	return data, labels
