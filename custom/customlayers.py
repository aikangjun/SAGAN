import numpy as np
import tensorflow as tf
from custom import *
from tensorflow.python.keras.utils import conv_utils


class ConvSelfAttention(layers.Layer):
    '''
    data_format:为'channels_last'或为‘channels_first’
    注意没有filters参数
    '''

    def __init__(self,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str,
                 data_format: str,
                 return_attention: bool,
                 kernel_initializer=initializers.GlorotUniform(),
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 gamma_initializer=initializers.Zeros(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ConvSelfAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.return_attention = return_attention
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
        config = super(ConvSelfAttention, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'return_attention': self.return_attention,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 4
        if self.data_format == 'channels_first':
            # input_shape:(batch_size,num_channels,height,width)
            channel_axis = 1
        else:
            # input_shape:(batch_size,height,width,num_channels)
            channel_axis = -1
        in_channels = input_shape[channel_axis]
        assert in_channels >= 8, '输入的通道数必须大于等于8'
        if not in_channels:
            raise ValueError('输入的channel维度应该被定义，但是发现为None')

        kernel_f_shape = self.kernel_size + (in_channels, in_channels // 8)
        kernel_g_shape = self.kernel_size + (in_channels, in_channels // 8)
        kernel_h_shape = self.kernel_size + (in_channels, in_channels)
        # kernel_o_shape 核的形状为什么有in_channels//2?
        #
        kernel_o_shape = self.kernel_size + (in_channels, in_channels)

        self.kernel_f = self.add_weight(name='kernel_f',
                                        shape=kernel_f_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        self.kernel_g = self.add_weight(name='kernel_g',
                                        shape=kernel_g_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        self.kernel_h = self.add_weight(name='kernel_h',
                                        shape=kernel_h_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        self.kernel_o = self.add_weight(name='kernal_o',
                                        shape=kernel_o_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=(),
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=True)
        self.built = True

    def hw_flatten(self, x):
        if self.data_format == 'channels_first':
            h, w = x.shape[2], x.shape[3]
            return tf.reshape(x, (tf.shape(x)[0], -1, h * w))
        else:
            h, w = x.shape[1], x.shape[2]
            return tf.reshape(x, (tf.shape(x)[0], h * w, -1))

    def transpose_matmul(self, x, y):
        if self.data_format == 'channels_first':
            return tf.matmul(tf.transpose(self.hw_flatten(x), perm=[0, 2, 1]), self.hw_flatten(y))
        else:
            return tf.matmul(self.hw_flatten(x), tf.transpose(self.hw_flatten(y), perm=[0, 2, 1]))

    def call(self, inputs, *args, **kwargs):
        bs = tf.shape(inputs)[0] # 必须使用tf.shape(),在创建动态图时，不这样使用将会报错，重要
        if self.data_format == 'channels_first':
            height, width = tf.shape(inputs)[2], tf.shape(inputs)[3]
        else:
            height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]

        f = tf.nn.conv2d(input=inputs, filters=self.kernel_f, strides=self.strides, padding=self.padding.upper(),
                         data_format=conv_utils.convert_data_format(self.data_format, ndim=4))
        f = tf.nn.max_pool2d(input=f, ksize=2, strides=self.strides, padding=self.padding.upper(),
                             data_format=conv_utils.convert_data_format(self.data_format, ndim=4))
        g = tf.nn.conv2d(inputs, self.kernel_g, self.strides,
                         self.padding.upper(), conv_utils.convert_data_format(self.data_format, ndim=4))
        h = tf.nn.conv2d(inputs, self.kernel_h, self.strides, self.padding.upper(),
                         conv_utils.convert_data_format(self.data_format, ndim=4))
        h = tf.nn.max_pool2d(h, ksize=2, strides=self.strides, padding=self.padding.upper(),
                             data_format=conv_utils.convert_data_format(self.data_format, 4))

        e = self.transpose_matmul(f, g)
        attention_map = tf.nn.softmax(e)

        if self.data_format == 'channels_first':
            o = tf.matmul(self.hw_flatten(h), attention_map, transpose_b=True)
            o = tf.reshape(o, shape=(bs, -1, height, width))
        else:
            o = tf.matmul(tf.transpose(self.hw_flatten(h), perm=[0, 2, 1]), attention_map)
            o = tf.reshape(o, shape=(bs, height, width, -1))
        o = tf.nn.conv2d(o, self.kernel_o, self.strides, self.padding.upper(),
                         conv_utils.convert_data_format(self.data_format, ndim=4))
        x = tf.add(inputs, o * self.gamma)

        if self.return_attention:
            return x, attention_map
        else:
            return x


class ConvSN2D(layers.Conv2D):
    '''
    使用Spectral Normalization的卷积层
    Spectral Normalization是一种wegiht Normalization技术，
    和weight-clipping以及gradient penalty一样，也是让模型满足1-Lipschitz条件的方式之一。
    '''

    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str,
                 data_format: str,
                 sn_initializer=initializers.RandomNormal(0, 1),
                 **kwargs):
        super(ConvSN2D, self).__init__(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding=padding,
                                       data_format=data_format,
                                       **kwargs)
        self.sn_initializer = sn_initializer

    def get_config(self):
        config = super(ConvSN2D, self).get_config()
        config.update({
            'sn_initializer': self.sn_initializer,
        })
        return config

    def build(self, input_shape):
        super(ConvSN2D, self).build(input_shape)
        self.sn = self.add_weight(name='sn',
                                  shape=(1, self.filters),
                                  initializer=self.sn_initializer,
                                  trainable=False)
        self.built = True

    def call(self, inputs):
        def _l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(tf.matmul(_u, tf.transpose(W, perm=[1, 0])))
            _u = _l2normalize(tf.matmul(_v, W))

            return _u, _v

        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.sn)
        # sigma代表谱范数
        sigma = tf.matmul(_v, W_reshaped)
        sigma = tf.matmul(sigma, tf.transpose(_u, perm=[1, 0]))
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([self.sn.assign(_u)]):
            W_bar = tf.reshape(W_bar, W_shape)

        output = tf.nn.conv2d(inputs, W_bar, self.strides, self.padding.upper(),
                              conv_utils.convert_data_format(self.data_format, ndim=4))
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias,
                                    conv_utils.convert_data_format(self.data_format, ndim=4))
        if self.activation is not None:
            return self.activation(output)
        return output


class ConvSN2DTranspose(layers.Conv2D):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str,
                 data_format: str,
                 output_padding=None,
                 dilation_rate: tuple = (1, 1),
                 activation: str = None,
                 use_bias: bool = True,
                 kernel_initializer=initializers.GlorotUniform(),
                 bias_initializer=initializers.Zeros(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 sn_initializer=initializers.RandomNormal(0, 1),
                 **kwargs):
        super(ConvSN2DTranspose, self).__init__(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding,
                                                data_format=data_format,
                                                dilation_rate=dilation_rate,
                                                activation=activation,
                                                use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                activity_regularizer=activity_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                bias_constraint=bias_constraint,
                                                **kwargs)
        self.sn_initializer = sn_initializer

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(self.output_padding,
                                                             2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad > stride:
                    raise ValueError('Strides must be greater than output padding. '
                                     f'Received strides={self.strides}, '
                                     f'output_padding={self.output_padding}.')

    def get_config(self):
        config = super(ConvSN2DTranspose, self).get_config()
        config.update({
            'sn_initializer': self.sn_initializer,
            'output_padding': self.output_padding
        })

    def build(self, input_shape):
        assert len(input_shape) == 4
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        in_channels = input_shape[channel_axis]

        if not in_channels:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        keras_shape = self.kernel_size + (self.filters, in_channels)
        self.kernel = self.add_weight(name='kernel',
                                      shape=keras_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)
        self.sn = self.add_weight(name='sn',
                                  shape=(1, in_channels),
                                  initializer=self.sn_initializer,
                                  trainable=False)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        dims = inputs.shape.as_list()
        height = dims[h_axis]
        width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        dilation_rate_y, dilation_rate_x = self.dilation_rate
        assert np.logical_and(np.greater_equal(dilation_rate_y, 1), np.greater_equal(dilation_rate_x, 1))

        # 获取反卷积特征形状
        if self.padding == 'same':
            out_height = height * stride_h
            out_width = height * stride_w
        elif self.padding == 'valid':
            if np.logical_and(np.equal(dilation_rate_y, 1), np.equal(dilation_rate_x, 1)) or not self.dilation_rate:
                out_height = height * stride_h + kernel_h - 1
                out_width = width * stride_w + kernel_w - 1
            elif np.logical_or(np.greater(dilation_rate_y, 1), np.greater(dilation_rate_x, 1)):
                out_height = height * stride_h + (kernel_h - 1) * dilation_rate_y
                out_width = width * stride_w + (kernel_w - 1) * dilation_rate_x
        else:
            raise ValueError("padding must be in the set {valid, same}")

        if self.data_format == 'channels_first':
            output_shape_tensor = tf.cast([batch_size, self.filters, out_height, out_width], dtype=tf.int32)
        else:
            output_shape_tensor = tf.cast([batch_size, out_height, out_width, self.filters], dtype=tf.int32)

        def _l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(tf.matmul(_u, tf.transpose(W, perm=[1, 0])))
            _u = _l2normalize(tf.matmul(_v, W))
            return _u, _v

        # 谱归一化
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.sn)
        sigma = tf.matmul(_v, W_reshaped)
        sigma = tf.matmul(sigma, tf.transpose(_u, perm=[1, 0]))
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([self.sn.assign(_u)]):
            W_bar = tf.reshape(W_bar, W_shape)

        outputs = tf.nn.conv2d_transpose(inputs,
                                         W_bar,
                                         output_shape_tensor,
                                         strides=self.strides,
                                         padding=self.padding.upper(),
                                         data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
                                         dilations=self.dilation_rate)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs,
                                     self.bias,
                                     data_format=conv_utils.convert_data_format(self.data_format, ndim=4))
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height = input_shape[h_axis]
        width = input_shape[w_axis]
        height = height if height is not None else input_shape[h_axis]
        width = width if width is not None else input_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        dilation_rate_y, dilation_rate_x = self.dilation_rate
        assert np.logical_and(np.greater_equal(dilation_rate_y, 1), np.greater_equal(dilation_rate_x, 1))

        if self.padding == 'same':
            out_height = height * stride_h
            out_width = height * stride_w
        elif self.padding == 'valid':
            if np.logical_and(np.equal(dilation_rate_y, 1), np.equal(dilation_rate_x, 1)) or not self.dilation_rate:
                out_height = height * stride_h + kernel_h - 1
                out_width = width * stride_w + kernel_w - 1
            elif np.logical_or(np.greater(dilation_rate_y, 1), np.greater(dilation_rate_x, 1)):
                out_height = height * stride_h + (kernel_h - 1) * dilation_rate_y
                out_width = width * stride_w + (kernel_w - 1) * dilation_rate_x
        else:
            raise ValueError("padding must be in the set {valid, same}")

        if self.data_format == 'channels_first':
            output_shape_tensor = [batch_size, self.filters, out_height, out_width]
        else:
            output_shape_tensor = [batch_size, out_height, out_width, self.filters]
        return output_shape_tensor


if __name__ == '__main__':
    convsn2d = ConvSN2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        data_format='channels_first')
    inps = tf.random.normal((4, 3, 12, 12))
    outps = convsn2d(inps)
    1
