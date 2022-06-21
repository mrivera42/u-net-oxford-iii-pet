import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# create model 
def EncoderBlock(inputs, n_filters,maxpool=True,skip=True):
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(n_filters,kernel_size=3,activation='relu',padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x,training=False)
    if maxpool == True:
        output = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    else:
        output = x
    skip_connection = x

    if skip == True:
        

        return output, skip_connection
    else:
        return output

def DecoderBlock(input, skip_connection, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters,3,strides=(2,2),padding='same')(input)
    x = tf.keras.layers.concatenate([x, skip_connection],axis=3)
    x = tf.keras.layers.Conv2D(n_filters,3,activation='relu',padding='same')(x)
    return x 

def UNet(input_size=(128,128,3),n_filters=32,n_classes=3):

    # Encoder 
    inputs = tf.keras.layers.Input(input_size)
    e1, skip1 = EncoderBlock(inputs,n_filters)
    e2, skip2 = EncoderBlock(e1,n_filters*2)
    e3, skip3 = EncoderBlock(e2,n_filters*4)
    e4, skip4 = EncoderBlock(e3,n_filters*8)
    e5 = EncoderBlock(e4,n_filters*16,maxpool=False,skip=False)

    # Decoder 
    d1 = DecoderBlock(e5,skip4,n_filters*8)
    d2 = DecoderBlock(d1,skip3,n_filters*4)
    d3 = DecoderBlock(d2,skip2,n_filters*2)
    d4 = DecoderBlock(d3,skip1,n_filters)

    x = tf.keras.layers.Conv2D(n_filters,3,activation='relu',padding='same')(d4)
    output = tf.keras.layers.Conv2D(n_classes,1,padding='same')(x)

    model = tf.keras.Model(inputs=inputs,outputs=output)
    return model 

model = UNet((128,128,3),32,3)
model._name = "U-Net"
model.summary()
tf.keras.utils.plot_model(model, to_file='figures/unet_plot.png', show_shapes=True, show_layer_names=True)

# load dataset 
# for file in os.listdir('oxford_pet/annotations/annotations/trimaps'):
#     print(file)

mask_path = 'oxford_pet/annotations/annotations/trimaps'
img_path = 'oxford_pet/images/images'
images = []
masks = []
for file in os.listdir(mask_path):
    masks.append(file)
for file in os.listdir(img_path):
    if '.mat' not in file: 
        images.append(file)
images.sort()
masks.sort()

# preprocessing 
x = []
y = []
for i in range(0,len(images)-1):
    img = cv2.imread(os.path.join(img_path,images[i]))
    if img is not None:
    
        img = cv2.resize(img,(128,128))
        img = np.reshape(img, (128,128,3))
        img = img / 255. 
        x.append(img)

        mask = cv2.imread(os.path.join(mask_path,masks[i]),0)
        
        mask = cv2.resize(mask,(128,128))
        mask = np.reshape(mask,(128,128,1))
        mask = mask - 1
        y.append(mask)

# split data into test, train, and validation
def split_dataset(x,y,train=0.8,val=0.1,test=0.1):
    length = len(x)
    train_length = int(len(x) * train)
    val_length = int(len(x) * val)
    x_train = x[:train_length]
    y_train = y[:train_length]
    x_val = x[train_length:train_length + val_length]
    y_val = y[train_length:train_length + val_length]
    x_test = x[train_length + val_length:]
    y_test = y[train_length + val_length:]

    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)

x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x,y)
print(f'x_train: {len(x_train)}')
print(f'y_train: {len(y_train)}')
print(f'x_val: {len(x_val)}')
print(f'y_val: {len(y_val)}')
print(f'x_test: {len(x_test)}')
print(f'y_test: {len(y_test)}')



# view data 
def display(x,y):
    img = x[0]
    mask = y[0]
    plt.subplot(1,2,1)
    plt.title("Orginal Image")
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title('Mask')
    plt.imshow(mask)
    plt.savefig('figures/unet_example')

display(x_train, y_train)

# training 

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

earlystop = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(x_train, y_train, batch_size=32,epochs=30,validation_data=(x_val,y_val),callbacks=[earlystop])

loss = pd.DataFrame(model.history.history)
fig = loss.plot()
fig.set_xlabel('Epoch')
fig.set_ylabel('Percent')
fig.set_title('Loss Plot')
fig = fig.get_figure()
fig.savefig('figures/unet_lossplot.png')

model.evaluate(x_test,y_test)
model.save('models/unet')



    





















