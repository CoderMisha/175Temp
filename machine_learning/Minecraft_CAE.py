import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, losses, optimizers
import PIL
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':
    #open the image files and normalize them between [0,1]
    #images are being read from a directory that contains input and target directories
    villagers = glob.iglob('Screenshots_Villagers_2000/Villager/*.png')
    noVillagers = glob.iglob('Screenshots_Villagers_2000/NoVillager/*.png')
    xdata = []
    ydata = []
    for f in villagers:
        with PIL.Image.open(f) as img:
            d = np.asarray(img).astype('float32') / 255.
            xdata.append(d)
    for f in noVillagers:
        with PIL.Image.open(f) as img:
            d = np.asarray(img).astype('float32') / 255.
            ydata.append(d)
    #split the data into training and test sets
    split = int(len(xdata)*0.75)
    xdata_train = xdata[:split]
    xdata_test = xdata[split:]
    ydata_train = ydata[:split]
    ydata_test = ydata[split:]

    #form the sets into tensors to feed to the model
    xdata_train = tf.reshape(xdata_train, (len(xdata_train),256,256,3))
    xdata_test = tf.reshape(xdata_test, (len(xdata_test),256,256,3))
    ydata_train = tf.reshape(ydata_train, (len(ydata_train),256,256,3))
    ydata_test = tf.reshape(ydata_test, (len(ydata_test),256,256,3))

    #create the model
    model = models.Sequential()

    #encoder
    model.add(layers.Conv2D(filters=8, kernel_size=2, strides=(2,2), padding='same', activation='relu', input_shape=(256,256,3)))
    model.add(layers.Conv2D(filters=16, kernel_size=2, strides=(2,2), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=2, strides=(2,2), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=2, strides=(2,2), padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.Dense(256))

    #decoder
    model.add(layers.Dense(1024))
    model.add(layers.Dense(32*16*16))
    model.add(layers.Reshape((16,16,32)))
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(filters=8, kernel_size=2, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(filters=3, kernel_size=2, strides=1, padding='same'))

    #compile the model and fit it to the training data
    model.compile(optimizers.Adam(learning_rate=0.001), loss=losses.MeanSquaredError(), metrics=['accuracy', 'mean_squared_error'])
    h = model.fit(xdata_train, ydata_train, epochs=1000, batch_size=32, verbose=0)

    #plot accuracy over the training epochs
    plt.plot(h.history['accuracy'], label='accuracy')
    #plt.plot(h.history['mean_squared_error'], label='mse')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([.5, 1])
    plt.legend(loc='lower right')
    plt.title('Training Accuracy')
    plt.savefig('accuracy_CAE.png')

    #save some image reconstructions
    fig,axes = plt.subplots(2, 3, sharey=True)
    axes[0][0].set_title('Input')
    axes[0][0].imshow(xdata_test[0])
    axes[0][1].set_title('Target')
    axes[0][1].imshow(ydata_test[0])
    axes[0][2].set_title('Output')
    axes[0][2].imshow(tf.reshape(model(tf.reshape(xdata_test[0], (1,256,256,3))), (256,256,3)))
    axes[1][0].imshow(xdata_test[1])
    axes[1][1].imshow(ydata_test[1])
    axes[1][2].imshow(tf.reshape(model(tf.reshape(xdata_test[1], (1,256,256,3))), (256,256,3)))
    plt.savefig('reconstructions_CAE.png')
