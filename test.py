from keras.models import load_model
from keras.datasets import mnist
import cv2

model = load_model('wts.h5')
img = cv2.imread('test.png')
resized = cv2.resize(img,(28,28))
cv2.imshow('w',resized)
cv2.waitKey()
cv2.destroyAllWindows()