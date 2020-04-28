import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Flatten,Dense,Input,Dropout
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

dataset = pd.read_csv('trainn.csv')
X = dataset.iloc[:,0].values
Y1 = dataset.iloc[:,1].values
Y2 = dataset.iloc[:,2].values
Y3 = dataset.iloc[:,3].values

img = np.zeros((X.shape[0],224,224,3))
for i in range(X.shape[0]):
    image = load_img(X[i], target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    img[i,:,:,:] = image  

Y1 = np.reshape(Y1, (-1,1))
onehotencoder = OneHotEncoder()
y1 = onehotencoder.fit_transform(Y1).toarray()
X_train, X_test, y_train, y_test = train_test_split(img, y1, test_size = 0.6)

model = VGG16(include_top=False, input_shape=(224,224,3))
for layer in model.layers:
	layer.trainable = False
flat1 = Flatten()(model.outputs)
layer = Dense(1024, activation = 'relu')(flat1)
layer = Dropout(0.4)(layer)
layer = Dense(128, activation = 'relu')(layer)
layer = Dropout(0.4)(layer)
output = Dense(3, activation = 'softmax')(layer)
model = Model(inputs=model.inputs, outputs=output)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size = 1, epochs = 36, validation_data = (X_test, y_test))
model.save('trained')

