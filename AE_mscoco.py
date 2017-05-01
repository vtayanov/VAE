from keras.datasets import mnist

from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.callbacks import History
from keras.datasets import cifar10
import dataset_download_2
history = History()

import numpy as np
import matplotlib.pyplot as plt
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#path = './dataset/training2'
#directories = dataset_download_2.search_dir(path)
#print(directories)
#x_train, x_test, x_center= dataset_download_2.load_data(directories)
#plt.imshow(x_test[0])
#plt.show()

path ='./dataset/training'
directories = dataset_download_2.search_dir(path)
print(directories)
x_train, x_train_cropped, x_center= dataset_download_2.load_data(directories)

path ='./dataset/testing'
directories = dataset_download_2.search_dir(path)
print(directories)
x_test,x_test_cropped, x_center= dataset_download_2.load_data(directories)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_test_cropped = x_test_cropped.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 32, 232, 3))
#x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

#noise_factor = 0.5
#x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=0.3, size=x_train.shape) 
#x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=0.3, size=x_test.shape) 

#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)



n = 5
plt.figure(figsize=(6, 20))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = Input(shape=(64, 64, 3))
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x=MaxPooling2D((2, 2), border_mode='same')(x)
encoded = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
#encoded=Dropout(0.25)(x) 

# at this point the representation is (32, 7, 7)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded= Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

from keras.utils.visualize_util import plot
plot(autoencoder, to_file='model.png',show_shapes=True, show_layer_names=True)
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

SVG(model_to_dot(autoencoder).create(prog='dot', format='svg'))

hist=autoencoder.fit(x_train, x_train,
                nb_epoch=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_cropped,x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

decoded_imgs = autoencoder.predict(x_test_cropped)
print decoded_imgs.shape
print decoded_imgs[0]
print decoded_imgs[1]
print decoded_imgs[2]

#reset the boundaries of an image

def image_reset(img1,img2):
    scale = 0.25
    mask = np.ones(img1.shape)
    l = int(scale * img1.shape[0]) - 1
    u = int((1.0 - scale) * img1.shape[0])
    mask[l:u, l:u, :] = 0.0
    cropped_part=img1*(1 - mask)
    original_cropped=img2*mask
    return cropped_part+original_cropped




n = 10
plt.figure(figsize=(10, 2))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(image_reset(decoded_imgs[i],x_test[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

