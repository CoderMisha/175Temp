import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, losses, optimizers
import PIL
import matplotlib.pyplot as plt
import glob, os, time
from IPython import display
import datetime

#modify memory growth on GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except Exception as e:
        print(e)

#globals
LAMBDA = 100
EPOCHS = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
log_dir="logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#normalize image pixels to [-1,1]
def normalize(input_img, real_img):
    input_img = (input_img / 127.5) - 1
    real_img = (real_img / 127.5) - 1
    return input_img, real_img

def downsample(filters, size=(2,2), norm=True):
    init = tf.random_normal_initializer(0.,0.02)
    out = models.Sequential()
    out.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=init, use_bias=False,
                         dtype='float32'))
    if norm: out.add(layers.BatchNormalization())
    out.add(layers.LeakyReLU())
    return out

def upsample(filters, size=(2,2), dropout=False):
    init = tf.random_normal_initializer(0., 0.02)
    out = models.Sequential()
    out.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=init, use_bias=False,
                                   dtype='float32'))
    out.add(layers.BatchNormalization())
    if dropout: out.add(layers.Dropout(0.3))
    out.add(layers.ReLU())
    return out

def Generator():
    ins = layers.Input((256,256,3))
    down=[
        downsample(64, 4, norm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4)
    ]
    up = [
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4)
    ]
    
    init = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=init, activation='tanh',
                                  dtype='float32')
    
    x = ins
    skips = []
    for d in down:
        x = d(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for u, skip in zip(up, skips):
        x = u(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return Model(inputs=ins, outputs=x)

def generator_loss(disc_out, gen_out, target):
    loss = loss_object(tf.ones_like(disc_out), disc_out)
    l1 = tf.reduce_mean(tf.abs(gen_out - target))
    total_loss = loss + l1*LAMBDA
    return total_loss, loss, l1

def Discriminator():
    init = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target')
    x = tf.keras.layers.concatenate([inp, tar])
    d1 = downsample(64, 4, False)(x)
    d2 = downsample(128, 4)(d1)
    d3 = downsample(256, 4)(d2)
    z1 = layers.ZeroPadding2D()(d3)
    c = layers.Conv2D(512, 4, strides=1, kernel_initializer=init, use_bias=False)(z1)
    b = layers.BatchNormalization()(c)
    lr = layers.LeakyReLU()(b)
    z2 = layers.ZeroPadding2D()(lr)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(z2)
    return Model(inputs=(inp,tar), outputs=last)

def discriminator_loss(disc_real, disc_gen):
    loss = loss_object(tf.ones_like(disc_real), disc_real)
    g_loss = loss_object(tf.zeros_like(disc_gen), disc_gen)
    total_loss = loss + g_loss
    return total_loss

def generate_img(model, inpt, target, name='training_images_GAN.png'):
    pred = model(inpt, training=True)
    disp = [inpt[0], target[0], pred[0]]
    titles = ['input', 'target', 'predicted']
    fig,axes = plt.subplots(1,3, figsize=(12.8,9.6))
    for i in range(3):
        axes[i].imshow(disp[i] * 0.5 + 0.5)
        axes[i].set_title(titles[i])
    plt.savefig(name)

#create the generator and discriminator
gen_opt = optimizers.Adam(1e-3, beta_1=0.5)
disc_opt = optimizers.Adam(1e-3, beta_1=0.5)
generator = Generator()
discriminator = Discriminator()

#assign the checkpoint directory
chkpt_dir = './training_chkpts'
prefix = os.path.join(chkpt_dir, 'chkpt')
checkpoint = tf.train.Checkpoint(generator_opt = gen_opt, discriminator_opt = disc_opt,
                             generator = generator, discriminator = discriminator)

#load checkpoint if needed
try:
    checkpoint.restore(tf.train.latest_checkpoint(chkpt_dir))
except:
    pass

@tf.function
def training_step(inpt, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_out = generator(inpt, training=True)

        disc_real_out = discriminator([inpt, target], training=True)
        disc_gen_out = discriminator([inpt, gen_out], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_gen_out, gen_out, target)
        disc_loss = discriminator_loss(disc_real_out, disc_gen_out)

    gen_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        display.clear_output(wait=True)
        for example_input, example_target in test_ds.take(1):
            generate_img(generator, example_input, example_target)
        print("Epoch: ", epoch)

        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            training_step(input_image, target, epoch)
        print()

        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix = prefix)
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    checkpoint.save(file_prefix = prefix)

if __name__ == '__main__':
    #load the image data
    villagers = glob.iglob('Screenshots_Villagers_2000/Villager/*.png')
    noVillagers = glob.iglob('Screenshots_Villagers_2000/NoVillager/*.png')
    xdata = []
    ydata = []
    for f in villagers:
        with PIL.Image.open(f) as img:
            d = (np.asarray(img).astype('float32')) / 127.5 - 1
            xdata.append(d)
    for f in noVillagers:
        with PIL.Image.open(f) as img:
            d = (np.asarray(img).astype('float32')) / 127.5 - 1
            ydata.append(d)
    split = int(len(xdata)*0.75)
    xdata_train = xdata[:split]
    xdata_test = xdata[split:]
    ydata_train = ydata[:split]
    ydata_test = ydata[split:]

    #create shuffled training and test datasets
    train_size = 200
    batch_size = 32
    test_size = 200
    train_dataset = tf.data.Dataset.from_tensor_slices((xdata_train, ydata_train))
    train_dataset = train_dataset.shuffle(train_size)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((xdata_test, ydata_test))
    test_dataset = test_dataset.batch(batch_size)

    #fit the model
    fit(train_dataset, EPOCHS, test_dataset)

    #example images
    for example_input, example_target in train_dataset.take(5):
            generate_img(generator, example_input, example_target, name='example_images_GAN.png')
