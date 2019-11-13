from keras import backend as K
from keras.layers import Layer

''' CUSTOM LOCALLY CONNECTED 1D TO ALLOW FOR PADDING
'''
class  LC1D(Layer):
    """Locally-connected layer for 1D inputs.
    The `LocallyConnected1D` layer works similarly to
    the `Conv1D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each different patch
    of the input.
    # Example
    ```python
        # apply a unshared weight convolution 1d of length 3 to a sequence with
        # 10 timesteps, with 64 output filters
        model = Sequential()
        model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
        # now model.output_shape == (None, 8, 64)
        # add a new conv1d on top
        model.add(LocallyConnected1D(32, 3))
        # now model.output_shape == (None, 6, 32)
    ```
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any `strides!=1` is incompatible with specifying
            any `dilation_rate!=1`.
        padding: Currently only supports `"valid"` (case-insensitive).
            `"same"` may be supported in the future.
        data_format: String, one of `channels_first`, `channels_last`.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`
    # Output shape
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    @interfaces.legacy_conv1d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LocallyConnected1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected1D '
                             '(only "valid" is supported): ' + padding)
        self.data_format = K.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        input_dim = input_shape[2]
        if input_dim is None:
            raise ValueError('Axis 2 of input should be fully-defined. '
                             'Found shape:', input_shape)
        output_length = conv_utils.conv_output_length(input_shape[1],
                                                      self.kernel_size[0],
                                                      self.padding,
                                                      self.strides[0])
        self.kernel_shape = (output_length,
                             self.kernel_size[0] * input_dim,
                             self.filters)
        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(output_length, self.filters),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=3, axes={2: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        length = conv_utils.conv_output_length(input_shape[1],
                                               self.kernel_size[0],
                                               self.padding,
                                               self.strides[0])
        return (input_shape[0], length, self.filters)

    def call(self, inputs):
        output = K.local_conv1d(inputs, self.kernel, self.kernel_size, self.strides)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(LocallyConnected1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
