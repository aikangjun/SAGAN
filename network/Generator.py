import tensorflow as tf
from network import *
from custom.customlayers import ConvSN2DTranspose, ConvSelfAttention


class Generator(layers.Layer):
    def __init__(self,
                 **kwargs):
        super(Generator, self).__init__(**kwargs)

    def build(self, input_shape):
        self.convblock1 = models.Sequential([ConvSN2DTranspose(512, (3, 3), (4, 4), 'same', 'channels_last'),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(0.1)])
        self.convblock2 = models.Sequential([ConvSN2DTranspose(256, (3, 3), (2, 2), 'same', 'channels_last'),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(0.1)])
        self.convblock3 = models.Sequential([ConvSN2DTranspose(128, (3, 3), (2, 2), 'same', 'channels_last'),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(0.1)])
        self.att1 = ConvSelfAttention((3, 3), (1, 1), 'same', 'channels_last', False)
        self.convblock4 = models.Sequential([ConvSN2DTranspose(64, (3, 3), (2, 2), 'same', 'channels_last'),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(0.1)])
        self.att2 = ConvSelfAttention((3, 3), (1, 1), 'same', 'channels_last', False)
        self.last = models.Sequential([ConvSN2DTranspose(3, (3, 3), (2, 2), 'same', 'channels_last'),
                                       layers.Activation('tanh')])

    def call(self, inputs, *args, **kwargs):
        in_shape = tf.shape(inputs)
        x = tf.reshape(inputs, (in_shape[0], 1, 1, inputs.shape[-1]))  # 在channels拿到维度，必须使用inputs.shape[-1]
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.att1(x)
        x = self.convblock4(x)
        x = self.att2(x)
        x = self.last(x)
        return x
