from network import *
from custom.customlayers import ConvSN2D, ConvSelfAttention
import tensorflow as tf


class Discriminator(layers.Layer):
    def __init__(self,
                 **kwargs):
        super(Discriminator, self).__init__(**kwargs)

    def build(self, input_shape):

        self.convblock1 = models.Sequential([ConvSN2D(64, (3, 3), (4, 4), 'same', 'channels_last'),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(0.1)])
        self.convblock2 = models.Sequential([ConvSN2D(128, (3, 3), (2, 2), 'same', 'channels_last'),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(0.1)])
        self.convblock3 = models.Sequential([ConvSN2D(128, (3, 3), (2, 2), 'same', 'channels_last'),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(0.1)])
        # self.att1 = ConvSelfAttention((3, 3), (1, 1), 'same', 'channels_last', False)
        self.convblock4 = models.Sequential([ConvSN2D(256, (3, 3), (2, 2), 'same', 'channels_last'),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(0.1)])
        # self.att2 = ConvSelfAttention((3, 3), (1, 1), 'same', 'channels_last', False)
        self.last = models.Sequential([ConvSN2D(1, (3, 3), (2, 2), 'same', 'channels_last'),
                                       layers.Activation('sigmoid')])

    def call(self, inputs, *args, **kwargs):
        # 判别器过强，不使用self-attention层
        x = self.convblock1(inputs)
        x = self.convblock2(x)
        x = self.convblock3(x)
        # x = self.att1(x)
        x = self.convblock4(x)
        # x = self.att2(x)
        x = self.last(x)
        x = tf.squeeze(x, axis=[1, 2])
        return x


if __name__ == '__main__':

    inps = tf.random.normal(shape=(4, 64, 64, 3))
    d = Discriminator()
    outps = d(inps)
    1
