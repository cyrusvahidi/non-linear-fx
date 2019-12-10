from tensorflow.keras.layers import Convolution1D, UpSampling1D
import tensorflow as tf

import tensorflow.math as tfm

# from keras.engine.topology import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose, Lambda, Layer
from tensorflow.keras.activations import softplus
from tensorflow.keras import regularizers

import math
import numpy as np

class DePool1D(UpSampling1D):
    '''Simplar to UpSample, yet traverse only maxpooled elements
    # Input shape
        3D tensor with shape:
        `(samples, channels, frames)` if dim_ordering='th'
        or 3D tensor with shape:
        `(samples, frames, channels)` if dim_ordering='tf'.
    # Output shape
        3D tensor with shape:
        `(samples, upsampled_frames, channels)` if dim_ordering='th'
        or 3D tensor with shape:
        `(samples, upsampled_frames, channels)` if dim_ordering='tf'.
    # Arguments
        size: one integers. The upsampling factor across time.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 3
    
    def __init__(self, pool1d_layer, *args, **kwargs):
        self._pool1d_layer = pool1d_layer
        super().__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = K.repeat_elements(X, self.size, axis=2)
#             output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(X, self.size, axis=1)
#             output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = T.grad(T.sum(self._pool1d_layer.get_output(train)), wrt=self._pool1d_layer.get_input(train)) * output

        return f
    
#     def get_config(self):
#         config = {
#             '_pool1d_layer': self._pool1d_layer
#         }
#         base_config = super(DePool1D, self).get_config()
        
#         return dict(list(base_config.items()) + list(config.items()))

class Deconvolution1D(Layer):
#     input_ndim = 4

    def __init__(self, binded_conv_layer, **kwargs):
        self._binded_conv_layer = binded_conv_layer
        self.W = self._binded_conv_layer

        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super(Deconvolution1D, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, x):
 
        output = K.conv2d(x, self.W, padding='valid')

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[2], 1)
    
class Conv1DLocal(Layer):
    def __init__(self, n_kernels=128, kernel_size=128, data_format='channels_last', 
                 kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01), kernel_constraint=None,
                 use_bias=True, bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), bias_constraint=None, 
                 **kwargs):
        self.rank               = 1
        self.n_kernels          = n_kernels
        self.kernel_size        = normalize_tuple(kernel_size, self.rank,'kernel_size')
        self.data_format        = data_format
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint  = kernel_constraint
        
        self.use_bias           = True
        self.bias_initializer   = bias_initializer
        self.bias_regularizer   = bias_regularizer
        self.bias_constraint    = bias_constraint
        
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super(Conv1DLocal, self).build(input_shape)  # Be sure to call this at the end

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size +  (1, 1)

        self.kernels = [0] * self.n_kernels
        self.bias    = [0] * self.n_kernels
        for i in range(self.n_kernels):
            self.kernels[i] = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel-{0}'.format(i),
                                          regularizer=self.kernel_regularizer)
            if self.use_bias:
                self.bias[i] = self.add_weight(shape=(1,),
                                            initializer=self.bias_initializer,
                                            name='bias-{0}'.format(i),
                                            regularizer=self.bias_regularizer)
            else:
                self.bias[i] = None
        # Set input spec.
#         self.input_spec = InputSpec(ndim=self.rank + 2,
#                                     axes={channel_axis: 1})
        
        self.built = True
        
    def call(self, x):
        X = [0] * 128
        for i in range(self.n_kernels):
            X[i] = softplus(K.conv1d(x[:,:,i:i+1], self.kernels[i], padding='same'))
            K.bias_add(
                X[i],
                self.bias[i],
                data_format=self.data_format)
        X = K.concatenate(X, axis=2)

        return X
    
#     def get_config(self):
#         config = {
#             'rank': self.rank,
#             'n_kernels': self.n_kernels,
#             'kernel_size': self.kernel_size,
#             'data_format': self.data_format,
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Conv1DLocal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_kernels)
    
class SAAF(Layer):

    def __init__(self, n_breakpoints=4, v_order=2, vw_init='random_uniform', vw_reg=regularizers.l2(0.01), batch=32, **kwargs):
        self.n_breakpoints = n_breakpoints
        self.v_order = v_order
        self.w_order = n_breakpoints - 1
        
        self.vw_init = vw_init
        self.vw_reg = vw_reg
                
        self.batch = batch
        
        super(SAAF, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SAAF, self).build(input_shape)  # Be sure to call this at the end
        param_shape = list((int(input_shape[1]), 1))

        self.breakpoints = [int(input_shape[1]) / self.w_order * i for i in range(self.n_breakpoints)]
#         if self.shared_axes is not None:
#             for i in self.shared_axes:
#                 param_shape[i - 1] = 1
#                 self.param_broadcast[i - 1] = True
        self.n_activations = input_shape[-1]
        self.v = [[0] * self.v_order] * self.n_activations
        self.w = [[0] * self.w_order] * self.n_activations
        
        for i in range(self.n_activations):
            for j in range(self.v_order):
                self.v[i][j] = self.add_weight(shape=param_shape,
                                             name='v-{0}-{1}'.format(i, j),
                                             initializer=self.vw_init,
                                             regularizer=self.vw_reg)
            for j in range(self.w_order):
                self.w[i][j] = self.add_weight(shape=param_shape,
                                             name='w-{0}-{1}'.format(i, j),
                                             initializer=self.vw_init,
                                             regularizer=self.vw_reg)
#         # Set input spec
#         axes = {}
#         if self.shared_axes:
#             for i in range(1, len(input_shape)):
#                 if i not in self.shared_axes:
#                     axes[i] = input_shape[i]

        self.built = True

    def call(self, x):
        
        X = []
        for i in range(self.n_activations):
            self.i = i
            frame = x[:, :, i:i+1]
            
            sigma1 = K.variable(np.zeros((self.batch, frame.shape[1], frame.shape[2])))
            for j in range(self.v_order):
                j_fact = math.factorial(j)
                p = tfm.divide(K.pow(frame, j), j_fact)
                sigma1 = tfm.add(sigma1, tfm.multiply(self.v[i][j], p))

            sigma2 = K.variable(np.zeros((self.batch, frame.shape[1], frame.shape[2])))
            for j in range(1, self.w_order + 1):
                self.k = j
                sigma2 = tfm.add(sigma2, K.map_fn(self.basis2, frame))
            
            X.append(tfm.add(sigma1, sigma2))
            
        output = K.concatenate(X, axis=2)
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def basis2(self, x):
#         b_2_k = K.variable(np.zeros(x.shape[0], ))
        
        k_1 = tf.constant(self.breakpoints[self.k - 1], dtype=tf.float32)
#         print(k_1.shape)
        k = tf.constant(self.breakpoints[self.k], dtype=tf.float32)
        
        def f1(): 
            return tfm.add(tfm.subtract(tfm.divide(tfm.multiply(x, x), 2), tfm.multiply(k_1, x)), 
                                          tfm.divide(tfm.multiply(k_1, k_1), 2))
        def f2(): 
            val1 = tfm.divide(tfm.multiply(tfm.subtract(k, k_1), tfm.subtract(k, k_1)), 2)
            val2 = tfm.multiply(tfm.subtract(k, k_1), tfm.subtract(x, k))
            val = tfm.add(val1, val2)
            return val
        
        b2ks = [0] * x.shape[0]
        for i in range(x.shape[0]):
            b2ks[i] = tf.cond(tfm.logical_and(tfm.greater(x[i, 0], k_1), tfm.less(x[i, 0], k)), lambda: f1(), lambda: f2())
#             print(b2ks[i].shape)
        
        b_2_k = K.concatenate(b2ks)
#         if tfm.greater(x, k_1) and tfm.less(x, k):

#         elif tfm.less(k, x):
        
        return tfm.multiply(self.w[self.i][self.k - 1], b_2_k)
    
class DenseLocal(Layer):
    def __init__(self, units=64, n_kernels=128,
                 kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01), kernel_constraint=None,
                 use_bias=True, bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), bias_constraint=None, 
                 **kwargs):
        self.units              = int(units)
        self.n_kernels          = n_kernels
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint  = kernel_constraint
        
        self.use_bias           = True
        self.bias_initializer   = bias_initializer
        self.bias_regularizer   = bias_regularizer
        self.bias_constraint    = bias_constraint
        
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        super(DenseLocal, self).build(input_shape)  # Be sure to call this at the end

        input_dim = input_shape[-1]

        self.kernels = [0] * self.n_kernels
        self.bias    = [0] * self.n_kernels
        for i in range(self.n_kernels):
            self.kernels[i] = self.add_weight(
                                    'kernel-{0}'.format(i),
                                    shape=(int(input_dim), self.units),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
            if self.use_bias:
                self.bias[i] = self.add_weight(
                                      'bias-{0}'.format(i),
                                      shape=(self.units,),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      trainable=True)
            else:
                self.bias[i] = None
                
        self.built = True
        
    def call(self, x):
        rank = len(x.shape)
        X = [0] * x.shape[1]
        
        for i in range(self.n_kernels):
          # Broadcasting is required for the inputs.
            X[i] = softplus(K.dot(x[:,i:i+1,:], self.kernels[i]))

            K.bias_add(
                X[i],
                self.bias[i],
                data_format='channels_last')
        X = K.concatenate(X, axis=1)

        return X
    
#     def get_config(self):
#         config = {
#             'units': self.units,
#             'n_kernels': self.n_kernels,
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(DenseLocal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

def normalize_tuple(value, n, name):
    """Transforms a single int or iterable of ints into an int tuple.
    # Arguments
        value: The value to validate and convert. Could be an int, or any iterable
          of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. `strides` or
          `kernel_size`. This is only used to format error messages.
    # Returns
        A tuple of n integers.
    # Raises
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `{}` argument must be a tuple of {} '
                             'integers. Received: {}'.format(name, n, value))
        if len(value_tuple) != n:
            raise ValueError('The `{}` argument must be a tuple of {} '
                             'integers. Received: {}'.format(name, n, value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `{}` argument must be a tuple of {} '
                                 'integers. Received: {} including element {} '
                                 'of type {}'.format(name, n, value, single_value,
                                                     type(single_value)))
    return value_tuple

class PTanh(Layer):
    def __init__(self, alpha_initializer='zeros', **kwargs):
        
        self.alpha_initializer = alpha_initializer
        
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        super(PTanh, self).build(input_shape)  # Be sure to call this at the end

        input_dim = input_shape[-1]
        
        self.n_activations = input_dim

        self.alphas = [0] * self.n_activations
        param_shape = list((input_shape[1], 1))
        
        for i in range(self.n_activations):
            self.alphas[i] = self.add_weight(
                                    'ptan-{0}'.format(i),
                                    shape=(1, ),
                                    initializer=self.alpha_initializer)
                
        self.built = True
        
    def call(self, x):
        rank = len(x.shape)
        X = [0] * x.shape[-1]
        
        for i in range(self.n_activations):
          # Broadcasting is required for the inputs.
            X[i] = tfm.tanh(tfm.multiply(self.alphas[i], x[:,:,i:i+1]))

        X = K.concatenate(X, axis=2)

        return X

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])