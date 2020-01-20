#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import keras 
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
import matplotlib as mpl
from keras.layers import Input, Dense
from keras.models import Model


# In[2]:


x_train = pickle.load(open("x_train.obj","rb"))
x_test = pickle.load(open("x_test.obj","rb"))
y_train = pickle.load(open("y_train.obj","rb"))


# ### Dataset Overview

# In[3]:


x_train.shape


# In[4]:


x_test.shape


# In[5]:


type(x_train)


# In[6]:


type(x_test)


# In[7]:


y_train.shape


# In[8]:


y_train[0]


# In[9]:


# Find the number of observations for each digit in the y_train dataset.
# Which is the most frequent class?

from collections import Counter
count = Counter(y_train)
count


# ### Data preparation

# In[10]:


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[11]:


x_train = x_train.reshape(14000, 784)
x_test = x_test.reshape(8800, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Put everything on grayscale
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train,27)


# In[12]:


y_train.shape


# In[13]:


y_train[0]


# In[14]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.2)


# In[15]:


x_train[0].shape


# In[16]:


plt.imshow(x_train[0].reshape(28, 28))


# In[17]:


print(np.asarray(range(26)))
print(y_train[0].astype('int'))


# ## Modelling using Keras

# In[18]:


import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.colors as mcolors


# In[19]:


from __future__ import absolute_import, division, print_function, unicode_literals

mpl.rcParams['figure.figsize'] = (18, 12)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ### Defining some utility functions
# 
# Le seguenti funzioni saranno utilizzate per visualizzare le performance dei modelli sulla base dell'andamento di **loss, AUC, Precision** e **Recall** nella progressione dell'apprendimento durante le differenti epoche

# In[20]:


def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], 
             color=colors[0],
             label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0],
             linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.2,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,100])
  plt.ylim([0,100])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')


# In[21]:


def plot_loss(history):
  metrics =  ['loss']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], 
             color=colors[0],
             label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0],
             linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()


# ### Defining some metrics

# In[22]:


METRICS = [ 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]


# ### Defining the Neural Network Model

# In[23]:


n_epochs = 70
batch_size = 1024

dims = x_train.shape[1]
nb_classes = 27

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)


# In[24]:


def make_model(metrics = METRICS):
  model = keras.Sequential([
      keras.layers.Dense(512, activation='relu',kernel_initializer='TruncatedNormal', input_shape=(dims,)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(512, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(nb_classes, activation='softmax'),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss='categorical_crossentropy',
      metrics=METRICS)

  return model


# ### Baseline Neural Network Model

# In[25]:


model = make_model()
model.summary()


# In[26]:


model_baseline = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks = [early_stopping],
    validation_data=(x_val, y_val))


# In[27]:


plot_metrics(model_baseline)


# ### Autoencoder 

# In[28]:


# size of encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

#input placeholder
input_img = Input(shape=(784,))

# "encoded" are the encoded representations of the input
encoded = Dense(128, activation='relu')(input_img)
encoded_1 = Dense(encoding_dim, activation='relu')(encoded)    

# "decoded" are the lossy reconstructions of the input
decoded = Dense(128, activation='relu')(encoded_1)
decoded_1 = Dense(784, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded_1)


# In[29]:


# this model maps an input to its encoded representation
encoder = Model(input_img, encoded_1)


# In[30]:


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-2](encoded_input)
decoder_layer_1 = autoencoder.layers[-1](decoder_layer)

# create the decoder model
decoder = Model(encoded_input, decoder_layer_1)


# In[31]:


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()


# In[32]:


autoencoder_model = autoencoder.fit(x_train, x_train,
                epochs=70,
                batch_size=256,
                shuffle=True,
                validation_data=(x_val, x_val))


# In[33]:


plot_loss(autoencoder_model)


# ### Visual Investigation of Autoencoder abilities

# In[34]:


encoded_imgs_train = encoder.predict(x_train)
decoded_imgs_train = decoder.predict(encoded_imgs_train)

encoded_imgs_val = encoder.predict(x_val)
decoded_imgs_val = decoder.predict(encoded_imgs_val)

plt.figure(figsize=(40, 4))
for i in range(10):
    
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_val[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display encoded image
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(encoded_imgs_val[i].reshape(8,4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, 20, 2*20 + i + 1)
    plt.imshow(decoded_imgs_val[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()


# ### Neural Network Model with encoded input

# In[35]:


#input dim = 32
def make_model_2(metrics = METRICS):
  model = keras.Sequential([
      keras.layers.Dense(512, activation='relu',kernel_initializer='TruncatedNormal', input_shape=(32,)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(512, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(nb_classes, activation='softmax'),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss='categorical_crossentropy',
      metrics=METRICS)

  return model


# In[36]:


model_encoded_input = make_model_2()
model_encoded_input.summary()


# In[37]:


model_encoded_input_hist = model_encoded_input.fit(
    encoded_imgs_train,
    y_train,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks = [early_stopping],
    validation_data=(encoded_imgs_val, y_val))


# In[38]:


plot_metrics(model_encoded_input_hist)


# ## Prediction on Test Set

# In[39]:


encoded_imgs_test = encoder.predict(x_test)


# In[40]:


probs =  model_encoded_input.predict(encoded_imgs_test)


# In[41]:


predictions = probs.argmax(axis=-1)


# In[42]:


np.savetxt('Raffaele_Anselmo_846842_Score2.txt', predictions)

