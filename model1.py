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
from keras.models import load_model

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

Y3 = np.reshape(Y3, (-1,1))
onehotencoder1 = OneHotEncoder()
y3 = onehotencoder1.fit_transform(Y3).toarray()
X_train, X_test, y_train, y_test = train_test_split(img, y3, test_size = 0.6)

#model = VGG16(include_top=False, input_shape=(224,224,3))
#for layer in model.layers:
#	layer.trainable = False
#flat1 = Flatten()(model.outputs)
#layer = Dense(1024, activation = 'relu')(flat1)
#layer = Dropout(0.4)(layer)
#layer = Dense(128, activation = 'relu')(layer)
#layer = Dropout(0.4)(layer)
#output = Dense(8, activation = 'softmax')(layer)
#model = Model(inputs=model.inputs, outputs=output)
model = load_model('trained12')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size = 1, epochs = 10, validation_data = (X_test, y_test))

model.save('trained12')


