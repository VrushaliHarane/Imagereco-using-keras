import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import OneHotEncoder


dataset = pd.read_csv('trainn.csv')
Y1 = dataset.iloc[:,1].values
Y2 = dataset.iloc[:,2].values
Y3 = dataset.iloc[:,3].values

Y1 = np.reshape(Y1, (-1,1))
onehotencoder = OneHotEncoder()
y1 = onehotencoder.fit_transform(Y1).toarray()

Y2 = np.reshape(Y2, (-1,1))
onehotencoder2 = OneHotEncoder()
y2 = onehotencoder2.fit_transform(Y2).toarray()

Y3 = np.reshape(Y3, (-1,1))
onehotencoder3 = OneHotEncoder()
y3 = onehotencoder3.fit_transform(Y3).toarray()


b_model = load_model('trained')
b_model3 = load_model('trained12')
b_model2 = load_model('trained2')
X = '/home/vrushali/images.jpeg'
image = load_img(X, target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, 224, 224, 3))
img = preprocess_input(image) 

var = b_model.predict(img)
var2 = b_model2.predict(img)
var3 = b_model3.predict(img)

var = onehotencoder.inverse_transform(var)
var2 = onehotencoder2.inverse_transform(var2)
var3 = onehotencoder3.inverse_transform(var3)
print(var) #gender
print(var2) #type
print(var3)  #colour




