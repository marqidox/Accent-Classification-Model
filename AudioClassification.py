#!/usr/bin/env python
# coding: utf-8

# # Tutorial Links
# https://github.com/krishnaik06/Audio-Classification/blob/main/Audio%20Classification%20EDA.ipynb https://github.com/krishnaik06/Audio-Classification/blob/main/Part%20--%20Audio%20Classification%20Data%20Preprocessing%20And%20Model%20Creation.ipynb

# In[1]:


import matplotlib.pyplot as plt
import librosa
import librosa.feature
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import gc


# In[2]:


#DATA EXPLORATION
a = 'C:'
b = 'Users'
c = 'hiloo'
d = 'Downloads'
e = 'archive'
f = 'recordings'
g = 'recordings'
h = 'spanish1.mp3'

filename = os.path.join(a + os.sep, b, c, d, e, f, g, h)

# plotting the audio
plt.figure(figsize=(14,5))
data,sr=librosa.load(filename)
librosa.display.waveshow(data,sr=sr)
plt.show()
gc.collect()


# In[3]:


# generate mel spectrogram
mel_sp = librosa.feature.melspectrogram(y=data, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel_sp, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.show()
gc.collect()


# In[2]:


# creating my dataframe
dict = {'filename': [],
        'class': []}

a = 'C:'
b = 'Users'
c = 'hiloo'
d = 'Downloads'
e = 'archive'
f = 'recordings'
g = 'recordings'

filedir = os.path.join(a + os.sep, b, c, d, e, f,g)
for root, dirs, files in os.walk(filedir):
    # getting file accent
    for file in files:
        name = file.split(".")[0]
        while not name.isalpha():
            name = name[:-1]
        dict['filename'].append(os.path.join(filedir,file))
        dict['class'].append(name)

df = pd.DataFrame(dict)
gc.collect()


# In[3]:


print(df)
gc.collect()


# AUDIO AUGMENTATIONS TO ARTIFICIALLY INFLATE DATA

# In[5]:


import audiomentations
from audiomentations import *

augment1 = audiomentations.Compose([
    AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.2, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-5, max_semitones=5, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

def apply_audio_transforms(file):
    y,sr = librosa.load(file)
    aug_y = augment1(y, sr)
    return aug_y


# In[6]:


df_aug = df.copy()
df_aug
gc.collect()


# In[7]:


from tqdm import tqdm
df_aug['filename'] = df_aug['filename'].progress_apply(apply_audio_transforms)
gc.collect()


# In[8]:


df_aug


# In[10]:


def get_y(file):
    y,sr = librosa.load(file)
    return y
    
df['filename'] = df['filename'].progress_apply(get_y)


# In[11]:


df = pd.concat([df,df_aug])
df


# In[1]:


df


# In[12]:


# shuffling dataset for randomness
# need to resample the classes so there is an even class
df = df.sample(frac=1, random_state=0, ignore_index=True)
print(df.head())
print(df.shape[0])
gc.collect()


# In[13]:


# creating feature extractor (Mel-Frequency Cepstral Coefficients(MFCC))
# summarizes frequency and time characteristics
from tqdm import tqdm
from librosa.util import fix_length
# for non-augmented data
def features_extractor(arr, aug=False):
    #audio = audio[:480000]
    pad_arr = fix_length(arr,size=30*22050)
    mfccs_features = librosa.feature.mfcc(y=pad_arr, sr=22050, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# In[15]:


df['feature'] = df['filename'].progress_apply(features_extractor)
df = df.drop(['filename'], axis=1)


# In[29]:


df


# In[11]:


df3 = df.copy()


# In[12]:


from collections import Counter
labels = [lang for lang, _ in Counter(df3['class']).most_common(2)]
df3 = df3[df3['class'].isin(labels)]
df3


# In[16]:


df3['class'].value_counts()


# In[53]:


df['class'].value_counts()


# In[23]:


import audiomentations
from audiomentations import *

noise = audiomentations.Compose([
    AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.2, p=1),
])
time_stretch = audiomentations.Compose([
    TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
])
pitch_shift = audiomentations.Compose([
    PitchShift(min_semitones=-5, max_semitones=5, p=1),
])
shift = audiomentations.Compose([
    Shift(min_shift=-0.5, max_shift=0.5, p=1),
])


def apply_audio_transforms1(file):
    y,sr = librosa.load(file)
    aug_y = noise(y, sr)
    return aug_y
def apply_audio_transforms2(file):
    y,sr = librosa.load(file)
    aug_y = time_stretch(y, sr)
    return aug_y
def apply_audio_transforms3(file):
    y,sr = librosa.load(file)
    aug_y = pitch_shift(y, sr)
    return aug_y
def apply_audio_transforms4(file):
    y,sr = librosa.load(file)
    aug_y = shift(y, sr)
    return aug_y


# In[17]:


# random undersampling of all but minority class
# Set the seed value for experiment reproducibility.
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')
y = df3[['class']]
df3, y_resampled = rus.fit_resample(df3, y)
del y
df3['label'] = y_resampled
del y_resampled
gc.collect()


# In[19]:


df3['class'].value_counts()


# In[20]:


from tqdm import tqdm
dn = df3.copy()
dts = df3.copy()
dps = df3.copy()
ds = df3.copy()
gc.collect()


# In[21]:


def get_y(file):
    y,sr = librosa.load(file)
    return y
    
df3['filename'] = df3['filename'].progress_apply(get_y)


# In[24]:


dn['filename'] = dn['filename'].progress_apply(apply_audio_transforms1)


# In[25]:


dts['filename'] = dts['filename'].progress_apply(apply_audio_transforms2)


# In[26]:


dps['filename'] = dps['filename'].progress_apply(apply_audio_transforms3)


# In[27]:


ds['filename'] = ds['filename'].progress_apply(apply_audio_transforms4)


# In[28]:


df4 = pd.concat([df3,dn,dts,dps,ds])
gc.collect()


# In[30]:


df4 = df4.drop(['label'], axis=1)
df4


# In[33]:


# creating feature extractor (Mel-Frequency Cepstral Coefficients(MFCC))
# summarizes frequency and time characteristics
from tqdm import tqdm
from librosa.util import fix_length
# for non-augmented data
def features_extractor(arr, aug=False):
    #audio = audio[:480000]
    pad_arr = fix_length(arr,size=30*22050)
    mfccs_features = librosa.feature.mfcc(y=pad_arr, sr=22050, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# In[34]:


df4['feature'] = df4['filename'].progress_apply(features_extractor)
df4 = df4.drop(['filename'], axis=1)


# In[35]:


# shuffling dataset for randomness
# need to resample the classes so there is an even class
df4 = df4.sample(frac=1, random_state=0, ignore_index=True)
print(df4.head())
print(df4.shape[0])
gc.collect()


# In[36]:


# split dataset into independent and dependent
X = np.array(df4['feature'].tolist())
y = np.array(df4['class'].tolist())

print(X.shape)
print(y)


# In[37]:


# label encoding
# used to convert categorical columns/labels into numerical ones
# so they can be used by machine learning programs
import tensorflow as tf
keras = tf.keras
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(y))


# In[38]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[39]:


# creating the model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from sklearn import metrics

# number of classes
num_labels = y.shape[1] # num of columns

# setting up model
model=Sequential()
#first layer
# input shape is my x_train last dimension
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()


# In[40]:


# categorical_crossentropy measures loss between true and predicted labels
# used for multiclass classification problems
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[41]:


# Training the model
from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 250
num_batch_size = 32

a = 'C:'
b = 'Users'
c = 'hiloo'
d = 'PycharmProjects'
e = 'audioClassificationSpectrograms'
f = 'saved_models'
g = 'accent_classification10_engspan.hdf5'

filepath = os.path.join(a + os.sep, b, c, d, e, f, g)
# saving my best model
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test,y_test), callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[42]:


# testing accuracy
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(test_accuracy[1])


# In[45]:


from librosa.util import fix_length
def features_extractor2(file, aug=False):
    #audio = audio[:480000]
    arr = get_y(file)
    pad_arr = fix_length(arr,size=30*22050)
    mfccs_features = librosa.feature.mfcc(y=pad_arr, sr=22050, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# In[51]:


# english
a = 'C:'
b = 'Users'
c = 'hiloo'
d = 'PycharmProjects'
e = 'audioClassificationSpectrograms'
f = 'personalizedAudioClassification'
g = 'recordings'
h = 'english'
i = 'english316.mp3'

filepath = os.path.join(a + os.sep, b, c, d, e, f, g, h, i)
prediction_feature=features_extractor2(filepath)
prediction_feature=prediction_feature.reshape(1,-1)
print(prediction_feature)
predict=model.predict(prediction_feature) 
print(predict)
pred_test = np.argmax(predict, axis=-1)
print(pred_test)
prediction_class = label_encoder.inverse_transform(pred_test)
prediction_class


# In[52]:


# spanish
a = 'C:'
b = 'Users'
c = 'hiloo'
d = 'PycharmProjects'
e = 'audioClassificationSpectrograms'
f = 'personalizedAudioClassification'
g = 'recordings'
h = 'spanish'
i = 'spanish74.mp3'

filepath = os.path.join(a + os.sep, b, c, d, e, f, g, h, i)
prediction_feature=features_extractor2(filepath)
prediction_feature=prediction_feature.reshape(1,-1)
print(prediction_feature)
predict=model.predict(prediction_feature) 
print(predict)
pred_test = np.argmax(predict, axis=-1)
print(pred_test)
prediction_class = label_encoder.inverse_transform(pred_test)
prediction_class


# In[50]:


y,sr = librosa.load(filepath)

#resample = librosa.resample(y,sr,8000)


# In[ ]:




