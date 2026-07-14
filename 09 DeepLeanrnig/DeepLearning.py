import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# np.set_printoptions(linewidth=115,precision=1)
# print(X_train[0])

# for i in range(10):
#     plt.subplot(1,10,i+1)
#     plt.imshow(X_train[i].reshape((28,28)),"gray")#imshow関数は画像(行列)をプロット
# plt.show()

import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.utils import plot_model

X_train = X_train.reshape(X_train.shape[0],784)[:6000]#X_train.shape[0]:画像枚数60000が入っている
X_test = X_test.reshape(X_test.shape[0],784)[:1000]
y_train = to_categorical(y_train)[:6000]#数字であった正解データを01配列に変換
y_test = to_categorical(y_test)[:1000]

inputs = Input(shape=(784,))
x = Dense(256)(inputs)
x = Activation('sigmoid')(x)
x = Dense(10)(x)
outputs = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="sgd",loss="categorical_crossentropy",
              metrics=["accuracy"])
plot_model(model,"m1.png",
           show_layer_names = False)
image = plt.imread("m1.png")
plt.figure(dpi=150)
plt.imshow(image)
plt.show()

# his = model.fit(X_train,y_train,epochs=5)

# plt.plot(his.history["accuracy"],label="acc",
#          marker = "o")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(loc="best")
# plt.show()