import math, time, random, datetime
import numpy, pandas
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
numpy.set_printoptions(threshold=numpy.inf)

train = pandas.read_csv('UNSW-NB15 Source-Files/UNSW-NB15 - CSV Files/a part of training and testing set/UNSW_NB15_training-set.csv')
test = pandas.read_csv('UNSW-NB15 Source-Files/UNSW-NB15 - CSV Files/a part of training and testing set/UNSW_NB15_testing-set.csv')
#print(test.head())
#print(train.head())

data = pandas.concat([train,test]).reset_index(drop=True)
cols_cat = data.select_dtypes('object').columns
cols_numeric = data._get_numeric_data().columns

#print(data.describe())
#print(data.isnull().sum())

def Remove_dump_values(data, cols):
    for col in cols:
        data[col] = numpy.where(data[col] == '-', 'other', data[col])
    return data

cols = data.columns
data_bin = Remove_dump_values(data, cols)
data_bin = data_bin.drop(['id'], axis=1) #Remove Unnecessary features
labels_cat = data_bin['attack_cat']
data_bin.drop(['attack_cat','label'], axis=1, inplace=True)
cols_cat = cols_cat.drop(['attack_cat'])

#print(data_bin.describe())
data_bin_hot = pandas.get_dummies(data_bin,columns=cols_cat)
#print(data_bin_hot.shape)

cols_numeric = list(cols_numeric)
cols_numeric.remove('label')
cols_numeric.remove('id')
data_bin_hot[cols_numeric] = data_bin_hot[cols_numeric].astype('float')
data_bin_hot[cols_numeric] = (data_bin_hot[cols_numeric] - numpy.min(data_bin_hot[cols_numeric])) / numpy.std(data_bin_hot[cols_numeric])

training_labels_hot = pandas.get_dummies(labels_cat)
training_data = data_bin_hot.to_numpy(dtype='float32')
training_labels = training_labels_hot.to_numpy(dtype='float32')

#print(training_data[:10])
#print(training_labels[:10])
# Data split

train_data = training_data[:175342]
train_labels = training_labels[:175342]
validation_data = training_data[175342:]
validation_labels = training_labels[175342:]

print('Starting TensorFlow.Keras....')
import keras
from keras import models, layers, regularizers, optimizers, losses, metrics

model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_shape=(len(train_data[0]),)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.categorical_crossentropy, metrics=[metrics.accuracy])

history = model.fit(train_data, train_labels, batch_size=256, epochs=100, validation_data=(validation_data, validation_labels))

# Plot for training and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

start_by_epoch = 1
epochs = range(start_by_epoch, len(loss_values) + 1)

plt.plot(epochs, loss_values[start_by_epoch-1:], 'bo', label='Training loss')
plt.plot(epochs, val_loss_values[start_by_epoch-1:], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()

# Plot for training and validation accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc[start_by_epoch-1:], 'bo', label='Training accuracy')
plt.plot(epochs, val_acc[start_by_epoch-1:], 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
plt.clf()

print('Evaluated model:')
print(model.evaluate(validation_data, validation_labels, batch_size=128))
predictions = model.predict(validation_data)
print(validation_labels[:5])
print(predictions[:5])

mode_name = ('bakalarska-praca-' + datetime.datetime.now().strftime('%d-%m-%Y@%H-%M-%S') + '.h5')
print('Saving model as ' + mode_name + '...')
model.save(mode_name)