import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]

# create dummy variables
geography = pd.get_dummies(X.Geography)
gender = pd.get_dummies(X.Gender)

# concat and drop
X = pd.concat([X,geography,gender],axis=1)
X = X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ------- ANN -----------
from keras.models import Sequential
from keras.layers import Dense,Dropout,LeakyReLU,ELU,PReLU

# instantiate model
classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 13))
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

classifier.compile(optimizer= 'adamax',loss='binary_crossentropy',metrics=['accuracy'])

model_history = classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=100)


# predicting
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


# list all data in history
print(model_history.history.keys())

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
