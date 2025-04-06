
import numpy as np
from tensorflow.python.keras.saving.saved_model.load import metrics

import cv2
import tensorflow as tf
from tensorflow import keras

# print(tf.__version__)    # Check TensorFlow version
# print(keras.__version__) # Check Keras version


(X_train, y_train) , (X_test, y_test)= keras.datasets.mnist.load_data()

img=cv2.resize(X_train[0],(400,400))
cv2.imshow('new',img)
cv2.waitKey(0)

# print(y_train[:5])

# SCALING
X_train= X_train/255
X_test=X_test/255

# FLATTENING
X_train_flat=X_train.reshape(len(X_train),28*28)
X_test_flat=X_test.reshape(len(X_test),28*28)




# model= keras.Sequential([
#     keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(X_train_flat,y_train,epochs=5)
#
# model.evaluate(X_test_flat,y_test)
#
#
# y_predict=model.predict(X_test_flat)
# n=5
# y_predict_real= [np.argmax(i) for i in y_predict]
#
# print(y_test[:5])
# print(y_predict_real[:5])
#
# cm = tf.math.confusion_matrix(labels=y_test,predcitions=y_predict_real)


model= keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,),activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flat,y_train,epochs=5)


model.evaluate(X_test_flat,y_test)
print(y_test[:5])
model.save("mnist_model.h5")

# y_predict=model.predict(X_test_flat)
# n=5
# y_predict_real= [np.argmax(i) for i in y_predict]
#
# print(y_predict_real[:5])

