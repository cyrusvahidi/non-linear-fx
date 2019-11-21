from tensorflow.keras.layers import Dense, Input, Activation, Conv1D, MaxPool1D, BatchNormalization, LocallyConnected1D, Dropout, UpSampling1D, Lambda, multiply, Reshape, ZeroPadding1D, Flatten, ELU, Permute, Concatenate, Conv2DTranspose, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2 

from model.layers import DePool1D, Deconvolution1D

import tensorflow as tf

class NonLinearFXModel():
    def __init__(self, params_data=None, params_train=None, dropout=True, dropout_rate=0.5):
        self.params_data = params_data
        self.params_train = params_train
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        
        self.build()
        
    def get_model(self):
        return self.model
    
    def build(self):
        input_shape = (self.params_data.get('frame_size'), 1)

        x = Input(shape=input_shape, name='input_frame')
        
        self.w1 = Conv1D(filters=128, kernel_size=64,
                   activation='linear',
                   padding='same',
                   kernel_initializer='random_uniform',
                   name='conv1')
        self.w2 = [0] * 128
        self.dense_local = [0] * 128
    
        for i in range(128):
            self.w2[i] = Conv1D(filters=1, kernel_size=128,
                   activation='softplus',
                   kernel_initializer='random_uniform',
                   name='conv2-{0}'.format(i))
            
            self.dense_local[i] = Dense(64,
                   kernel_initializer='random_uniform',
                   kernel_regularizer=l2(1e-3),
                   activation='softplus',
                   name='dnn-local-{0}'.format(i))

            
        self.mp = MaxPool1D(pool_size=16, name='max_pool')
        
        z = self.frontend(x)
        z = self.latent_dnn(z)
        y = self.backend(z)
        
        self.model = Model(inputs=x, outputs=y)

        return self.model
    
    def frontend(self, x):
        # L1 128x64 filters, absolute activation
        x1 = self.w1(x)
        self.X1 = x1
        x = Lambda(lambda t: K.abs(t), name="abs_activation")(x1)

        # L2 LOCALLY CONNECTED, 128 x 128 filters, softplus activation
        x = ZeroPadding1D((64,63))(x)
        X2 = [0] * 128
        for i in range(128):
            activ = Lambda(lambda x: x[:,:,i:i+1], output_shape=(self.params_train.get('batch_size'), 1151, 1))(x) 
            X2[i] = self.w2[i](activ) 
        X2 = Concatenate(axis=2)(X2)
        
        x = BatchNormalization()(X2)
        x = self.mp(x) # apply max pooling layer
        
        return x
    
    def backend(self, x):
        #     x = UpSampling1D(size=16)(x)
        x = DePool1D(self.mp, size=16, name='depool')(x)

        # residual connection - element-wise multiply residual and X2
        x = multiply([x, self.X1])

        x = self.dnn_saaf(x)

        x = ZeroPadding1D((32, 31))(x) # zero pad the time series before doing 1d convolution
#         y = Deconvolution1D(self.w1)(x)
        x = Permute((2, 1), input_shape=(1024, 128))(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=3))(x) # add dimension to input for 1d conv
        
        kernel = Lambda(lambda x: K.transpose(x))(self.w1.kernel) # transpose kernel
        kernel = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(kernel) # switch kernel dimensions up
        kernel = Lambda(lambda x: K.expand_dims(x, axis=3))(kernel) # add dimension to the kernel
       
        x = Deconvolution1D(kernel)(x)
#         x = Lambda(lambda x: K.conv2d(x[0], x[1], padding='valid'))([x, kernel])
        
        x = Lambda(lambda x: K.squeeze(x, axis=3))(x) # remove extra dimension
        x = Permute((2, 1), input_shape=(1, 1024))(x) # permute output dims
     
        return x
        
    def dnn_saaf(self, x):
        # dnn saaf - 4 FC LAYERS 128 64, 64 - SOFTPLUS, 128 - locally connected SAAF
        x = Dense(128,
              kernel_initializer='random_uniform',
              kernel_regularizer=l2(1e-3),
              activation='softplus',
              name='saaf-fc1')(x)
        if self.dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Dense(64,
              kernel_initializer='random_uniform',
              kernel_regularizer=l2(1e-3),
              activation='softplus',
              name='saaf-fc2')(x)
        if self.dropout: 
            x = Dropout(self.dropout_rate)(x)

        x = Dense(64,
              kernel_initializer='random_uniform',
              kernel_regularizer=l2(1e-3),
              activation='softplus',
              name='saaf-fc3')(x)
        if self.dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Dense(128,
              kernel_initializer='random_uniform',
              kernel_regularizer=l2(1e-3),
              name='saaf-fc4')(x)
        x = ELU()(x) # TODO: locally connected SAAF activation

        if self.dropout:
            x = Dropout(self.dropout_rate)(x)
        
        return x

    def latent_dnn(self, z):
        z = Permute((2, 1), input_shape=(64, 128))(z)
        Z = [0] * 128
        for i in range(128):
            activ = Lambda(lambda x: x[:,i:i+1,:], output_shape=(self.params_train.get('batch_size'), 1, 64))(z) 
            Z[i] = self.dense_local[i](activ) 
        Z = Concatenate(axis=1)(Z)
        if self.dropout:
            Z = Dropout(self.dropout_rate)(Z)
            
        Z = Dense(64,
              kernel_initializer='random_uniform',
              kernel_regularizer=l2(1e-3),
              activation='softplus',
              name='dnn-2')(Z)
        if self.dropout:
            Z = Dropout(self.dropout_rate)(Z)
            
        Z = Permute((2, 1), input_shape=(128, 64))(Z)

        return Z
