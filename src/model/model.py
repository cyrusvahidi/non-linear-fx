from tensorflow.keras.layers import Dense, Input, Activation, Conv1D, MaxPool1D, BatchNormalization, LocallyConnected1D, Dropout, UpSampling1D
from tensorflow.keras import backend as K

import tensorflow as tf


def unsupervised_model(params_data=None, params_train=None):
    input_shape = (params_data.get('frame_size'), 1)
    
    x = Input(shape=input_shape, name='input_frame')
    w1 = Conv1D(filters=128, kernel_size=64,
               activation='linear',
               kernel_initializer='random_uniform',
               name='conv1')
    w2 = LocallyConnected1D(filters=128, kernel_size=128,
               activation='softplus',
               kernel_initializer='random_uniform',
               name='conv2')

    # L1 128x64 filters, absolute activation
    x1 = w1(x)
    x = Lambda(lambda t: K.abs(t), name="abs_activation")(x1)
    
    # L2 LOCALLY CONNECTED, 128 x 128 filters, softplus activation
    x2 = w2(x)
    x = BatchNormalization()(x2)
    x = MaxPool1D(pool_size=16)(x)
    
    # TODO: Unpool using locations
    x = UpSampling1D(size=16)(x)
    # residual connection - element-wise multiply residual and X2
    x = K.multiply(x, x1)
    
    # dnn saaf - 4 FC LAYERS 128 64, 64 - SOFTPLUS, 128 - locally connected SAAF
    x = Dense(128,
          kernel_initializer='random_uniform',
          kernel_regularizer=l2(1e-3),
          activation='softplus',
          name='saaf-fc1')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64,
          kernel_initializer='random_uniform',
          kernel_regularizer=l2(1e-3),
          activation='softplus',
          name='saaf-fc2')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64,
          kernel_initializer='random_uniform',
          kernel_regularizer=l2(1e-3),
          activation='softplus',
          name='saaf-fc3')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128,
          kernel_initializer='random_uniform',
          kernel_regularizer=l2(1e-3),
          name='saaf-fc4')(x)
    
    # TODO: locally connected SAAF activation
    x = Dropout(0.5)(x)
    
    # DECONV, TRANSPOSE FIRST LAYER TRANSFORM - how to use w1 weights
#     y = tf.nn.conv1d_transpose(x, w1, name='deconv')
    y = w1(x)
    
    model = Model(inputs=x, outputs=y)

    return model

def get_dense_layers(z):
    z = Dense(64,
          kernel_initializer='random_uniform',
          kernel_regularizer=l2(1e-3),
          activation='softplus',
          name='dnn-1')(z)
    z = Dropout(0.5)(z)
    z = Dense(64,
          kernel_initializer='random_uniform',
          kernel_regularizer=l2(1e-3),
          activation='softplus',
          name='dnn-2')(z)
    z = Dropout(0.5)(z)
    
    return z
