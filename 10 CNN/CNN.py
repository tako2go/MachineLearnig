import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Activation, Dense, Input
from keras.models import Sequential, load_model
from keras.utils import to_categorical,plot_model
from keras.models import Model

np.random.seed(1)
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout
import time

import tensorflow as tf
tf.keras.utils.set_random_seed(42)

from keras.datasets import mnist
(X_train,y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)[:6000]
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

# inputs = Input(shape=(28,28,1))
# x = Conv2D(8, (3, 3),padding='same')(inputs)
# x = Activation('relu')(x)
# x = Flatten()(x)
# x = Dense(10)(x)
# outputs = Activation('softmax')(x)
# model = Model(inputs=inputs, outputs=outputs)

# model.compile(optimizer="sgd",
#               loss = "categorical_crossentropy",
#               metrics=["accuracy"])

# plot_model(model,"m1.png",
#            show_layer_names = False)
# image = plt.imread("m1.png")
# plt.figure(dpi=150)
# plt.imshow(image)
# plt.show()

# startTime = time.time()
# his = model.fit(X_train,y_train, epochs=20,
#                 validation_data = (X_test, y_test))
# print("computation time:{0:.3f} sec".format(time.time() - startTime))


# plt.plot(his.history["accuracy"],label="acc",
#          marker = "o")
# plt.plot(his.history["val_accuracy"],label="acc",
#          marker = "o",color="orange")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(loc="best")
# plt.show()


# def show_prediction():
#   y = model.predict(X_test)
#   plt.figure(2, figsize=(12,8))
#   for i in range(96):
#     plt.subplot(8,12,i+1)
#     x = X_test[i,:].reshape(28,28)
#     plt.imshow(x, "gray_r")
#     prediction = np.argmax(y[i,:])
#     plt.text(22,25, '%d' %prediction, fontsize=10)
#     if prediction != np.argmax(y_test[i,:]):
#       plt.plot([0,27],[1,1],color='cornflowerblue',linewidth=5)
#     plt.xticks([],"")
#     plt.yticks([],"")

# show_prediction()
# plt.show()





inputs = Input(shape=(28,28,1))

x = Conv2D(16, (3, 3),padding='same')(inputs)
x = Activation('relu')(x)

x = Conv2D(32, (3, 3))(x)
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3))(x)
x = Activation('relu')(x)

x = MaxPooling2D(2,2)(x)

x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(128)(x)
x = Activation('relu')(x)

x = Dropout(0.25)(x)

x = Dense(10)(x)
outputs = Activation('softmax')(x)
model2 = Model(inputs=inputs, outputs=outputs)

model2.compile(optimizer="sgd",
              loss = "categorical_crossentropy",
              metrics=["accuracy"])


startTime = time.time()
his = model2.fit(X_train,y_train, epochs=20,
                validation_data = (X_test, y_test))
print("computation time:{0:.3f} sec".format(time.time() - startTime))

plt.plot(his.history["accuracy"],label="acc",
         marker = "o")
plt.plot(his.history["val_accuracy"],label="acc",
         marker = "o",color="orange")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()