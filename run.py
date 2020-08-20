from tensorflow.keras.layers import Conv2D,Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np

model = Sequential()

model.add(Conv2D(24,(5,5),activation='relu',input_shape=(200,200,3),kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(36,(5,5),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(48,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))

model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))

model.add(Dense(1,activation='tanh',kernel_regularizer=regularizers.l2(0.01)))

model.summary()

model.summary()
model.load_weights("sdc2.h5")

img_str = cv2.imread('steering_wheel_image copy.jpg',0)
rows,cols = img_str.shape

smoothed_angle = 0

f = open("07012018/data.txt")
labels = f.readlines()
f.close()

i = 60000
while( cv2.waitKey(1) & 0xFF != ord('q') and i<70000):
    img = cv2.imread("07012018/data/"+str(i)+".jpg")
    cv2.imshow("frame", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img = cv2.resize(img,(200,200))
    img = np.array(img)/255.0
    
    img = np.reshape(img,[1,200,200,3])
    degrees = model.predict(img)*180/np.pi
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) *(degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img_str,M,(cols,rows))

    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
