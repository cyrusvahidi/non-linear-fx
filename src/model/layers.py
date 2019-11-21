from tensorflow.keras.layers import Convolution1D, UpSampling1D
import tensorflow as tf

# from keras.engine.topology import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose, Lambda, Layer

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

class Deconvolution1D(Layer):
#     input_ndim = 4

    def __init__(self, binded_conv_layer, **kwargs):
        self._binded_conv_layer = binded_conv_layer
#         kwargs['filters'] = self._binded_conv_layer.filters
#         kwargs['kernel_size'] = self._binded_conv_layer.kernel_size
#         self.W = self._binded_conv_layer.kernel
        self.W = self._binded_conv_layer

        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super(Deconvolution1D, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, x):
 
        output = K.conv2d(x, self.W, padding='valid')

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[2], 1)
    
#     def get_output(self, train=False):
#         X = self.get_input(train)
  
#         output = K.conv2d(X, self.W, padding='valid')
# #         X = K.permute_dimensions(X, (2,1))
# #         X = K.expand_dims(X, axis=3)
        
# #         filters = K.transpose(self.W)
# #         filters = K.permute_dimensions(filters, (0,2,1))
        
# #         output = K.conv2d(X, K.expand_dims(filters, axis=3), padding='valid')
# #         output = K.squeeze(output, axis=3)
# #         output = K.permute_dimensions(output, (2, 1))
        
#         return output

#     def get_config(self):
#         config = {
#             'filters': self.filters,
#             'kernel_size': self.kernel_size,
#             'strides': self.strides,
#             'padding': self.padding,
#             'activation': activations.serialize(self.activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer':
#                 regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(LocallyConnected1D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

# class SAAF(Layer):

#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(SAAF, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(SAAF, self).build(input_shape)  # Be sure to call this at the end

#     def call(self, x):
#         return K.dot(x, self.kernel)

#     def compute_output_shape(self, input_shape):