from tensorflow.keras.layers import Convolution1D, UpSampling1D

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

def Conv1DTranspose(input_tensor, filters, kernel_size, weights, strides=1, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x.set_weights(weights)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x