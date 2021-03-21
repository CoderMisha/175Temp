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
    out.add(SpectralConv2D(filters, size, strides=2, padding='same', kernel_initializer=init, use_bias=False,
                         dtype='float32'))
    if norm: out.add(layers.BatchNormalization())
    out.add(layers.LeakyReLU())
    return out

def upsample(filters, size=(2,2), dropout=False):
    init = tf.random_normal_initializer(0., 0.02)
    out = models.Sequential()
    out.add(layers.SpectralConv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=init, use_bias=False,
                                   dtype='float32'))
    out.add(layers.BatchNormalization())
    if dropout: out.add(layers.Dropout(0.3))
    out.add(layers.ReLU())
    return out

class AttentionModel(Model):
    def __init__(self, filters, **kwargs):
        super(AttentionModel, self).__init__(**kwargs)
        self.attention_layer = attention()
        self.filters = filters
        
    def call(self, inputs):
        return self.attention_layer(inputs, self.filters)

class attention(layers.Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(), initializer=tf.initializers.zeros)
        
    def call(self, x, filters):
        size = tf.shape(x)
        batch_size = size[0]
        height = size[1]
        width = size[2]
        channels = size[3]

        f = layers.Conv2D(filters // 8, 1, strides=1)(x)  # [bs, h, w, c']
        #f = layers.MaxPool2D(pool_size=(1,1))(f)

        g = layers.Conv2D(filters // 8, 1, strides=1)(x)  # [bs, h, w, c']

        h = layers.Conv2D(filters, 1, strides=1)(x)  # [bs, h, w, c]
        #h = layers.MaxPool2D(pool_size=(1,1))(h)

        proj_query = tf.reshape(f, (batch_size, height*width, -1))
        proj_key = tf.transpose(tf.reshape(g, (batch_size, height*width, -1)), (0, 2, 1))
        proj_value = tf.transpose(tf.reshape(h, (batch_size, height*width, -1)), (0, 2, 1))

        s = tf.matmul(proj_query, proj_key)
        print(s.shape)
        attn = tf.nn.softmax(s)
        print(attn.shape)
        out = tf.matmul(proj_value, tf.transpose(attn, (0, 2, 1)))
        print(out.shape)
        out = tf.reshape(tf.transpose(out, (0, 2, 1)), (batch_size, height, width, -1))
        print(out.shape)

        return tf.add(tf.multiply(out, self.gamma), x), attn

def L2Norm(inpt):
    return inpt / (tf.norm(inpt) + 1e-8)

class L2NormRandNorm(tf.initializers.TruncatedNormal):
    def __init__(self):
        super(L2NormRandNorm, self).__init__(mean=0.0, stddev=1.0)
    
    def __call__(self, shape, dtype=tf.dtypes.float32):
        start = super(L2NormRandNorm, self).__call__(shape, dtype=dtype)
        return L2Norm(start)

class SpectralConv2D(layers.Conv2D):
    def __init__(self, filters, kernel_size, iterations=1, **kwargs):
        super(SpectralConv2D, self).__init__(filters, kernel_size, **kwargs)
        self.iterations = iterations
        
    def build(self, input_shape):
        super(SpectralConv2D, self).build(input_shape)
        indim = input_shape[-1]
        kshape = self.kernel_size + (indim, self.filters)
        self.u = self.add_weight(shape=(self.filters,1), initializer=L2NormRandNorm, trainable=False)
        self.v = self.add_weight(shape=(self.kernel_size[0]*self.kernel_size[1]*indim, 1),
                                initializer=L2NormRandNorm, trainable=False)
    def spectral_norm(self):
        k = tf.reshape(self.kernel, (-1, self.filters))
        for _ in range(self.iterations):
            nv = L2Norm(tf.matmul(k, self.u))
            nu = L2Norm(tf.matmul(tf.transpose(k), self.v))
        print(nu.shape)
        print(nv.shape)
        print(k.shape)
        print(self.kernel.shape)
        s = tf.matmul(tf.matmul(tf.transpose(nv), k), nu)
        nk = tf.divide(self.kernel, s)
        print(nk.shape)
        return nu, nv, nk
    
    def call(self, inputs):
        nu,nv,nk = self.spectral_norm()
        out = self._convolution_op(inputs, nk)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias, data_format='NHWC')
        if self.activation is not None:
            out = self.activation(out)
        return out #+ [nu,nv,nk]

class SpectralConv2DTranspose (SpectralConv2D, layers.Conv2DTranspose):
    def __init__(self, filters, kernel_size, power_iterations=1, **kwargs):
        super(SpectralConv2DTranspose, self).__init__(filters, kernel_size, power_iterations=1, **kwargs)
    def call(self, inputs):
        nu,nv,nk = self.spectral_norm()
        height, width = inputs_shape[1], inputs_shape[2]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        out_height = conv_utils.deconv_output_length(height, kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width, kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        output_shape = (batch_size, out_height, out_width, self.filters)
        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.keras.backend.conv2d_transpose(inputs, new_kernel, output_shape_tensor,
            strides=self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias,
                                     data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs #+ [new_v, new_u, new_kernel]

def Generator():
    ins = layers.Input((256,256,3))
    down=[
        downsample(64, 4, norm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        AttentionModel(),
        downsample(512, 4),
        downsample(512, 4),
        AttentionModel(),
        downsample(512, 4),
        downsample(512, 4)
    ]
    up = [
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        AttentionModel()
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
    d3 = AttentionModel()(d2)
    d4 = downsample(256, 4)(d3)
    z1 = layers.ZeroPadding2D()(d4)
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
