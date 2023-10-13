Code for segmentation of words
# Program to do Page Segmentation of Telugu Documents
import numpy as np
import cv2
import os
path1="/home/saikrishna/projects/majorproject/project"
path2="/home/saikrishna/projects/majorproject/outputs/connected/both"
path3="/home/saikrishna/projects/majorproject/outputs/connected/separate/text"
path4="/home/saikrishna/projects/majorproject/outputs/connected/separate/images"
#threshhold th for horizontal segmentation
th=10
for fn in os.listdir(path1):
name=path1+'/'+fn
img=cv2.imread(name,0) # To read input as img
# implement RLSA Algorithm here.
imgarr=np.array(img)
print(imgarr[5])
#inversion and binarisation
for i in range(0,len(imgarr)):
for j in range(0,len(imgarr[i])):
if imgarr[i][j]<=127:
imgarr[i][j]=255
else:
imgarr[i][j]=0
print("after inversion")
print(imgarr[5])
height,width=img.shape[ :2]
print("height1 is"+str(height))
print("height2 is"+str(width))
i=0
j=0
# horizontal smearing
while (i<height):
print("hi1")
j=0
count=0
revcount=0
while j<width:
#print("hi2")
if(imgarr[i][j]==0):
count+=1


j+=1
print("i is "+str(i)+"j is "+str(j)+"count is "+str(count))
else:
revcount=0
print("value at ij is"+str(imgarr[i][j]))
if(count<=th and count!=0):
print("before(j+count)"+str(j))
while(revcount<count):
j-=1
imgarr[i][j]=255
revcount+=1
j=j+count
count=0
print("after(j+count)"+str(j))
j+=1
i=i+1
#vertical smearing
th=20
# 20 for telugu
# 5 for english
i=0
j=0
while (i<width):
print("hi1")
j=0
count=0
revcount=0
while j<height:
#print("hi2")
if(imgarr[j][i]==0):
count+=1
j+=1
print("i is "+str(i)+"j is "+str(j)+"cou nt is "+str(count))
else:
revcount=0
print("value at ij is"+str(imgarr[j][i]))
if(count<=th and count!=0):
print("before(j+count)"+str(j))
while(revcount<count):
j-=1
imgarr[j][i]=255
revcount+=1
j=j+count

count=0
print("after(j+count)"+str(j))
j+=1
i=i+1
#Connected component analysis
output = cv2.connectedComponentsWithStats(imgarr,8,cv2.CV_32S)
print(type(output[0]))
print(output[0])
print(type(output[1]))
print(output[1])
print(type(output[2]))
print(output[2])
stats = output[2]
#print(output[2][53])
print(type(output[3]))
print(output[3])
labels=output[1]
print(stats[0][3])
print(stats[0][1])
print(stats[0][2])
#writing separate components in separate folders
for i in range(0,output[0]):
x_cor=stats[i][0]
y_cor=stats[i][1]
height=y_cor+stats[i][3]
width=x_cor+stats[i][2]
img10 = img[y_cor:height,x_cor:width]
##to remove noise taking 500as area
area=stats[i][4]
if(i==0):
continue
if(area>500):

if(stats[i][3]<80):
#print(img10)
print("hi")
name3 = path3+'/'+str(i)+'.tiff'
cv2.imwrite(name3,img10)
else:
print("hi")
name3 = path4+'/'+str(i)+'.tiff'
cv2.imwrite(name3,img10)
Code for fitting the model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
start_time = time.time()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
batch_size = 50 #16 #128
num_classes = 47 #10
epochs =12 #12
#d is a dictionary for labels
d={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'13':13,'14':14,'15':
15,'16':16,'17':17,'18':18,'19':19,'20':20,'21':21,'22':22,'23':23,'24':24,'25':25,'26':26,'27':27,'2
8':28,'29':29,'30':30,'31':31,'32':32,'33':33,'34':34,'35':35,'36':36,'37':37,'38':38,'39':39,'40':40
,'41':41,'42':42,'43':43,'44':44,'45':45,'46':46}
# input image dimensions
img_rows, img_cols = 60,180
#training data
file = open('./data.pkl','rb')
x_train,y_train = pickle.load(file)
file.close()
T1=len(y_train)
for i in range(0,T1):

y_train[i]=d[y_train[i]]
# Splitting training data set into "train and test = 90% and 10% respectively"
random_seed = 2
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.1,
random_state=random_seed)
# Splitting 90% training data set into "train and validation = 80% and 10% respectively"
X_train, X_val, Y_train, Y_val= train_test_split(X_train, Y_train, test_size=0.1,
random_state=random_seed)
# Mapping to "list" into "array"
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_val = np.asarray(X_val)
Y_val = np.asarray(Y_val)
if K.image_data_format() == 'channels_first':
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)
else:
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255
print('X_train shape:', X_train.shape)
print(Y_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_val = keras.utils.to_categorical(Y_val, num_classes)
# Saving test data into a Separate file " test_data.pkl"
file = open('./test_data.pkl','wb')

pickle.dump((X_test,Y_test),file)
file.close()
#creating the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])
history=model.fit(X_train, Y_train,
batch_size=batch_size,
epochs=epochs,
verbose=1,validation_data=(X_val, Y_val))
score = model.evaluate(X_val, Y_val, verbose=0)
#generating the plots
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
model.summary()
# Here, it is the command to save the model, can loaded later for testing.
model.save('my_model.h5')
#model = load_model('my_model.h5')
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
print("--- %s seconds ---" % (time.time() - start_time))
print("--- strart time is %s seconds ---" % (start_time))
print("--- end time is %s seconds ---" % (time.time() ))

Code for Visualizing the matrix of first convolutional layer
''' Visualize the matrix of first convolutional layer'''
import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import models
import pandas as pd
import numpy as np
import itertools
# plotting figures
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
model = load_model('my_model.h5')
layer1 = model.layers[0]
layer1.name
conv2d_1w = layer1.get_weights()[0][:,:,0,:]
for i in range(1,33):
plt.subplot(8,4,i)
plt.imshow(conv2d_1w[:,:,i-1],interpolation="nearest",cmap="gray")
plt.show()
print("-----------------")
print("the values of the filters are ")
for i in range(1,33):
print(conv2d_1w[:,:,i-1])

Code for Visualizing the output of convolutional and pooling layers
import keras
#from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import time
start_time = time.time()
# modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,
BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import models
# plotting figures
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
model = load_model('my_model12.h5')
num_classes = 47
d={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'13':13,'14':14,'15
':15,'16':16,'17':17,'18':18,'19':19,'20':20,'21':21,'22':22,'23':23,'24':24,'25':25,'26':26,'27':2
7,'28':28,'29':29,'30':30,'31':31,'32':32,'33':33,'34':34,'35':35,'36':36,'37':37,'38':38,'39':39,'
40':40,'41':41,'42':42,'43':43,'44':44,'45':45,'46':46}
img_rows, img_cols = 60,180
file = open('./test_data.pkl','rb')

X_test,Y_test = pickle.load(file)
file.close()
T1=len(Y_test)
#2 graph representation
g = sns.countplot(Y_test)
Y_test_value=Y_test ## keep the original label
# Mapping to "list" into "array"
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
if K.image_data_format() == 'channels_first':
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)
else:
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_test = X_test.astype('float32')
X_test /= 255
print('X_test shape:', X_test.shape)
print(Y_test.shape)
print(X_test.shape[0], 'test samples')
Y_test = keras.utils.to_categorical(Y_test, num_classes)
test_im = X_test[2]
plt.imshow(test_im.reshape(img_rows, img_cols), cmap='viridis', interpolation='none')
# activation output
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(input=model.input, output=layer_outputs)
activations = activation_model.predict(test_im.reshape(1,img_rows, img_cols,1))
layer_names = []
print("layers are")
for layer in model.layers:

layer_names.append(layer.name)
print(layer.name)
print(len(layer_names))
images_per_row = 4
#code of 5.2 is on commenets below that 5.3 is present
for layer_name, layer_activation in zip(layer_names[:4], activations):
#if layer_name.startswith('conv'):
print(layer_activation)
n_features = layer_activation.shape[-1]
print(n_features)
size = layer_activation.shape[1]
size1 = layer_activation.shape[2]
#print(layer_activation.shape[2])
print(layer_activation.shape)
print("line105")
#size1 = layer_activation.shape[2]
print("line107")
print(layer_activation.shape[0])
print("size is "+str(size))
n_cols = n_features // images_per_row
display_grid = np.zeros((size * n_cols, images_per_row * size1)) ## * size
print(len(display_grid))
for col in range(n_cols):
for row in range(images_per_row):
channel_image = layer_activation[0,:, :, col * images_per_row + row]
print("len of channle image is")
print (len(channel_image))
l = len(channel_image)
len1= len(channel_image[0])
for i in range(0,l):
print(len(channel_image[i]))
channel_image -= channel_image.mean()
#print(channel_image)
channel_image /= channel_image.std()
#print(channel_image)
channel_image *=64
#print(channel_image)
channel_image += 128
channel_image = np.clip(channel_image, 0, 255).astype('uint8')
display_grid[col * size : (col + 1) * size,

row * size1 :(row + 1) * size1] = channel_image
scale = 1 / size
plt.figure(figsize=(scale * display_grid.shape[1],
scale * display_grid.shape[0]))
plt.title(layer_name, fontsize = 20)
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')
Code for Visualizing the output of fully connected layer
fc_layer = model.layers[-3]
activation_model = models.Model(input=model.input, output=fc_layer.output)
activations = activation_model.predict(test_im.reshape(1,60,180,1))
print("activations are")
print(len(activations[0]))
print(activations[0])
activation = activations[0].reshape(16,8) ### activations[0]len is 128 so toook 16,8
#if model.layers[-4] is take n then 157696=448,352
plt.imshow(activation, aspect='auto', cmap='viridis')
# organize the testing images by label
Y_test_value_df = pd.DataFrame(Y_test_value,columns=['label'])
Y_test_value_df['pos']=Y_test_value_df.index
Y_test_label_pos = Y_test_value_df.groupby('label')['pos'].apply(list)
pos = Y_test_label_pos[1][0]
#display 3 rows of digit image [0,9], with last full connected layer at bottom
plt.figure(figsize=(16,8))
x, y = 10, 1
for i in range(y):
for j in range(x):
# word image
plt.subplot(y*2, x, i*2*x+j+1)
pos = Y_test_label_pos[j][i] # j is label, i in the index of the list
plt.imshow(X_test[pos].reshape((img_rows, img_cols)),interpolation='nearest')
plt.axis('off')
plt.subplot(y*2, x, (i*2+1)*x+j+1)
activations = activation_model.predict(X_test[pos].reshape(1,img_rows, img_cols,1))
activation = activations[0].reshape(16,8)
plt.imshow(activation, aspect='auto', cmap='viridis')
plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()