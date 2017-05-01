'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import keras
from scipy import ndimage
import matplotlib.pyplot as plt
import keras.layers.advanced_activations
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Dropout
from keras.layers import Convolution2D, Deconvolution2D, BatchNormalization
from keras.models import Model
from keras.callbacks import History
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import objectives
from keras import optimizers as opt
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.datasets import mnist
import dataset_download_3
import dataset_download_2
import cv2
import matplotlib.gridspec as gridspec

def ImageReset(img1,img2):
    scale = 0.25
    mask = np.ones(img1.shape)
    l = int(scale * img1.shape[0]) - 1
    u = int((1.0 - scale) * img1.shape[0])
    mask[l:u, l:u, :] = 0.0
    cropped_part=img1*(1 - mask)
    original_cropped=img2*mask
    return cropped_part+original_cropped

#import multi_gpu
history = History()
# input image dimensions
img_rows, img_cols, img_chns = 64, 64, 3
# number of convolutional filters to use
nb_filters = 64
# convolution kernel size
nb_conv = 3
batch_size = 128

if K.image_dim_ordering() == 'th':
 original_img_size = (img_chns, img_rows, img_cols)
else:
 original_img_size = (img_rows, img_cols, img_chns)

latent_dim = 1000
intermediate_dim = 128
epsilon_std = 1.0
nb_epoch = 100

x = Input(
        batch_shape=(batch_size,) + original_img_size
        )


conv_1 = BatchNormalization()(Convolution2D(
        img_chns, 
        2,
        2, border_mode='same',
        activation='relu')(x))

conv_2 = BatchNormalization()( Convolution2D(
        nb_filters, 2, 2,
        border_mode='same',
        activation='relu',
        subsample=(2, 2)
        )(conv_1))

conv_3 = BatchNormalization()(Convolution2D(
        nb_filters, nb_conv, nb_conv,
        border_mode='same',
        activation='relu',
        subsample=(1, 1)
        )(conv_2))

conv_4 = BatchNormalization()(Convolution2D(
        nb_filters, nb_conv, nb_conv,
        border_mode='same',
        activation='relu',
        subsample=(1, 1)
        )(conv_3))

flat = Flatten()(conv_4)
hidden = BatchNormalization()(Dense(
        intermediate_dim,
        activation='relu'
        )(flat))

z_mean = Dense(latent_dim)(hidden)
z_log_var =Dense(latent_dim)(hidden)


def sampling(args):
 z_mean, z_log_var = args
 epsilon = K.random_normal(
         shape=(batch_size, latent_dim),
         mean=0., std=epsilon_std)
 return 0.5*(z_mean + K.exp(z_log_var) * epsilon)+0.5*(0.5*z_mean+K.exp(z_log_var) * epsilon)

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`

z = Lambda(
   sampling, 
   output_shape=(latent_dim,)
   )([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(
        intermediate_dim, 
        activation='relu'
        )


decoder_upsample = Dense(
        nb_filters * 32 * 32, 
        activation='relu'
        )

if K.image_dim_ordering() == 'th':
 output_shape = (
         batch_size, 
         nb_filters, 32, 32
         )
else:
 output_shape = (
         batch_size, 
         32, 
         32, 
         nb_filters
         )

decoder_reshape = Reshape(
        output_shape[1:]
        )

decoder_deconv_1 = Deconvolution2D(
        nb_filters, 
        nb_conv, 
        nb_conv,
        output_shape,
        border_mode='same',
        subsample=(1, 1),
        activation='relu'
        )

decoder_deconv_2 =Deconvolution2D(
        nb_filters, 
        nb_conv, nb_conv,
        output_shape,
        border_mode='same',
        subsample=(1, 1),
        activation='relu'
        )

if K.image_dim_ordering() == 'th':
 output_shape = (
         batch_size, 
         nb_filters, img_rows+1, 
         img_cols+1
         )
else:
 output_shape = (
         batch_size, 
         img_rows+1, 
         img_cols+1, 
         nb_filters
         )

decoder_deconv_3_upsamp = Deconvolution2D(
        nb_filters, 2, 2,
        output_shape,
        border_mode='valid',
        subsample=(2, 2),
        activation='relu'
        )

decoder_mean_squash = Convolution2D(
        img_chns, 2, 2,
        border_mode='valid',
        activation='relu'
        )

hid_decoded = decoder_hid(z)


up_decoded = decoder_upsample(hid_decoded)

reshape_decoded = decoder_reshape(up_decoded)

deconv_1_decoded = decoder_deconv_1(reshape_decoded)

deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)

x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)

x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

def vae_loss(x, x_decoded_mean):

# NOTE: binary_crossentropy expects a batch_size by dim
# for x and x_decoded_mean, so we MUST flatten these!
 
 x = K.flatten(x)
 
 x_decoded_mean = K.flatten(x_decoded_mean)
 
 xent_loss = img_rows * img_cols * objectives.binary_crossentropy(
         x, 
         x_decoded_mean)
 
 kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - 
            K.exp(z_log_var), axis=-1)
 return xent_loss+kl_loss

vae=(Model(x, x_decoded_mean_squash))
#vae=multi_gpu.make_parallel(vae, 2)

#Set optimisers
opt.SGD(
        lr = 0.001, 
        decay = 0.1, 
        momentum = 0.9, 
        nesterov = True
        )
opt.rmsprop(
        lr=0.001, 
        rho=0.9,
        epsilon=1e-08, 
        decay=0.0
        )
opt.adagrad(
        lr=0.01, 
        epsilon=1e-08, 
        decay=0.0
        )

opt.adadelta(
        lr=1.0, 
        rho=0.95, 
        epsilon=1e-08, 
        decay=0.0
        )

opt.adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08, 
        decay=0.0
        )

opt.adamax(
        lr=0.002, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-08, 
        decay=0.0
        )

opt.nadam(
        lr=0.0002, 
        beta_1=0.9,
        beta_2=0.999, 
        epsilon=1e-08, 
        schedule_decay=0.004
        )


#vae = make_parallel(vae, 4)
# train the VAE

path ='./dataset/training_batches/1'
directories = dataset_download_2.search_dir(path)
print(directories)
x_train, x_train_cropped, x_train_center= dataset_download_2.load_data(directories)

path ='./dataset/test_batches/1'
directories = dataset_download_2.search_dir(path)
print(directories)
x_test,x_test_cropped, x_test_center= dataset_download_2.load_data(directories)

print x_test.shape

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)

x_train_cropped = x_train_cropped.astype('float32') / 255.
x_train_cropped = x_train_cropped.reshape((x_train.shape[0],) + original_img_size)


x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

x_test_cropped = x_test_cropped.astype('float32') / 255.
x_test_cropped = x_test_cropped.reshape((x_test.shape[0],) + original_img_size)

noise_factor = 0.5
x_train_cropped_noisy = x_train_cropped + noise_factor * np.random.normal(
         loc=0.0, 
         scale=0.5, 
         size=x_train.shape
         ) 
x_test_cropped_noisy = x_test_cropped + noise_factor * np.random.normal(
       loc=0.0, 
       scale=0.5, 
       size=x_test.shape) 

x_train_cropped_noisy = np.clip(x_train_cropped_noisy, 0., 1.)
x_test_cropped_noisy = np.clip(x_test_cropped_noisy, 0., 1.)

#for i in range(0, 9):
#	plt.subplot(330 + 1 + i)
#	plt.imshow(x_test_cropped[i])
## show the plot
#plt.show()

fig = plt.figure()
gs = gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)

ax = [plt.subplot(gs[i]) for i in range(10*10)]

 
for i in range(100):
    ax[i].imshow(x_test_cropped[i])
    ax[i].axis('off')
plt.show()



plt.imshow(x_test_center[1])
plt.show()

vae.compile(
        optimizer='rmsprop', 
        loss=vae_loss
        )

vae.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

hist=vae.fit(
        x_train_cropped, x_train,
        verbose=1, 
        callbacks=[early_stopping],
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test_cropped, x_test)
        )
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

from keras.utils.visualize_util import plot

plot(
     encoder, 
     to_file='model_3.png',
     show_shapes=True, 
     show_layer_names=True
     )

# display a 2D plot of the images in the latent space
x_test_encoded = encoder.predict(x_test_cropped, batch_size=batch_size)
#plt.figure(figsize=(6, 6))

#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()

# build an image generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)

generator = Model(decoder_input, _x_decoded_mean_squash)


plot(
     generator, 
     to_file='model_2.png',
     show_shapes=True, 
     show_layer_names=True
     )

# display a 2D manifold of the images

n = 15 # figure with 15x15 digits
digit_size = 64
figure = np.zeros((digit_size * n, digit_size * n))

# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
#for i, yi in enumerate(grid_x):
 #for j, xi in enumerate(grid_y):
  #z_sample = np.array([[xi, yi]])
  #z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
  #x_decoded = generator.predict(z_sample, batch_size=batch_size)
  #np.swapaxes(x_decoded, 1,3)
  #np.swapaxes(x_decoded, 1,2)
  #print x_decoded.shape
  #print x_decoded[0][0]
  #digit=x_decoded[0]
 # digit = x_decoded[0].reshape(digit_size, digit_size,3)
 # figure[i * digit_size: (i + 1) * digit_size,
  #j * digit_size: (j + 1) * digit_size] = digit
#plt.figure(digit,figsize=(10, 10))


x_decoded = generator.predict(x_test_encoded, batch_size=batch_size)


#plt.imshow(x_decoded[1])
#plt.show()
cv2.imwrite("./dataset/generated/gen_result_0.png", x_decoded[0])

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
    plt.imshow(ImageReset(x_decoded[i],x_test[i]))
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


