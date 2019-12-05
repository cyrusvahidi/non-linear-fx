import os
import time
import datetime

from meta import *
from utils import load_audio, get_input_target_fname_pairs
from model.DataGenerator import DataGenerator
from model.NonLinearFXModel import NonLinearFXModel

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.models import Model

import pandas as pd

from sklearn.model_selection import train_test_split

def load_audio_pairs(instrument, fx, param_setting, input_target_file_pairs):
    n_total_clips = len(input_target_file_pairs)
    input_target_pairs =  [0] * n_total_clips
    for idx, (f_input, f_target) in enumerate(input_target_file_pairs):
        audio_in = load_audio(os.path.join(params_path[instrument][NO_FX], f_input), idx, n_total_clips, NO_FX)
        audio_target = load_audio(os.path.join(params_path[instrument][fx], f_target), idx, n_total_clips, fx)

        input_target_pairs[idx] = (audio_in, audio_target)
    return input_target_pairs

def save_test_data(input_target_file_pairs, out_path):
    # partition and save test data
    train_file_pairs, test_file_pairs = train_test_split(input_target_file_pairs,
                                           test_size=params_train.get('test_split'),
                                           random_state=42)
    df = pd.DataFrame(test_file_pairs, columns =['input_file', 'target_file']) 
    df.to_csv(os.path.join(out_path, 'test_data.csv'))
    
    return train_file_pairs

def load_data(instrument, fx, param_setting, out_path):
    input_target_file_pairs = get_input_target_fname_pairs(instrument, fx, param_setting)
    
    train_file_pairs = save_test_data(input_target_file_pairs, out_path)

    input_target_pairs = load_audio_pairs(instrument, fx, param_setting, train_file_pairs)

    train_pairs, val_pairs = train_test_split(input_target_pairs,
                                           test_size=params_train.get('val_split'),
                                           random_state=42)

    return train_pairs, val_pairs

def save_model(model, out_path, out_name):
    model.save_weights(os.path.join(out_path, '{0}.h5'.format(out_name)))

    # Save the model architecture
    with open(os.path.join(out_path, '{0}.json'.format(out_name)), 'w') as f:
        f.write(model.to_json())
        
def train_model(model, train_data_gen, val_data_gen):
    start = time.time()

    now = datetime.datetime.now()
    print("Current date and time:")
    print(str(now))
    early_stop = EarlyStopping(monitor='val_loss', patience=params_train.get('patience'), min_delta=0.001, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    callback_list = [early_stop, reduce_lr]

    hist = model.fit_generator(train_data_gen,
                   steps_per_epoch=params_train.get('n_iterations'), #train_data_gen.n_iterations,
                   epochs=params_train.get('n_epochs'),
                   validation_data=val_data_gen,
                   validation_steps=val_data_gen.n_iterations,
                   class_weight=None,
                   workers=0,
                   verbose=1,
                   callbacks=callback_list)

    end = time.time()
    print('\n=============================Job finalized==========================================================\n')
    print('Time elapsed for the job: %7.2f hours' % ((end - start) / 3600.0))
    print('\n====================================================================================================\n')
    return hist

def train_unsupervised(instrument, fx, param_setting, dropout, out_path):
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
    sess = tf.Session(config=config) 
    K.set_session(sess)

#     input_target_file_pairs = get_input_target_fname_pairs(instrument, fx, param_setting)
    
#     # partition and save test data
#     input_target_file_pairs, test_file_pairs = train_test_split(input_target_file_pairs,
#                                            test_size=params_train.get('test_split'),
#                                            random_state=42)
#     df = pd.DataFrame(test_file_pairs, columns =['input_file', 'target_file']) 
#     df.to_csv(os.path.join(out_path, 'test_data.csv'))
# #     import pdb; pdb.set_trace()
    
#     ## PUT AUDIO Loading into a function - do it in data_loader?
#     n_total_clips = len(input_target_file_pairs)
#     input_target_pairs =  [0] * n_total_clips
#     for idx, (f_input, f_target) in enumerate(input_target_file_pairs):
#         audio_in = load_audio(os.path.join(params_path[instrument][NO_FX], f_input), idx, n_total_clips, NO_FX)
#         audio_target = load_audio(os.path.join(params_path[instrument][fx], f_target), idx, n_total_clips, fx)

#         input_target_pairs[idx] = (audio_in, audio_target)

#     train_pairs, val_pairs = train_test_split(input_target_pairs,
#                                            test_size=params_train.get('val_split'),
#                                            random_state=42)

    train_pairs, val_pairs = load_data(instrument, fx, param_setting, out_path)

    # Generate input target frame pairs
    train_data_gen = DataGenerator(train_pairs, floatx=np.float32, batch_size=params_train['batch'], frame_size=params_data['frame_size'], hop_size=params_data['hop_size'], unsupervised=True)
    val_data_gen = DataGenerator(val_pairs, floatx=np.float32, batch_size=params_train['batch'], frame_size=params_data['frame_size'], hop_size=params_data['hop_size'], unsupervised=True, train=False)

    tr_loss, val_loss = [0] * params_train.get('n_epochs'), [0] * params_train.get('n_epochs')

    nlfx = NonLinearFXModel(params_data=params_data, params_train=params_train, dropout=dropout, dropout_rate=0.3)
    model = nlfx.build()
    opt = Adam(lr=params_train['lr'])
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    model.summary()

    hist = train_model(model, train_data_gen, val_data_gen)

    save_model(model, out_path, 'unsupervised_model')
    return model, nlfx, train_pairs, val_pairs

def train_supervised(model, nlfx, train_pairs, val_pairs):
    
    def get_layer_index(model, layer_name):
        index = None
        for idx, layer in enumerate(model.layers):
            if layer.name == layer_name:
                index = idx
                break
        return index

    # insert latent DNN layers into trained model
    layers = [l for l in model.layers]
    idx = get_layer_index(model, 'max_pool')
    latent_dnn = nlfx.latent_dnn(model.layers[idx].output)
    x = latent_dnn
    for i in range(idx + 1, len(layers)):
        if layers[i].name.startswith('multiply'): # handle case where we hit multiply layer 
            x = layers[i]([x, model.get_layer('conv1').output])
        else:
            x = layers[i](x)

    model = Model(inputs=layers[0].input, outputs=x)
    opt = Adam(lr=params_train['lr'])
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    model.summary()
    
        # Generate input target frame pairs
    train_data_gen = DataGenerator(train_pairs, floatx=np.float32, batch_size=params_train['batch'], frame_size=params_data['frame_size'], hop_size=params_data['hop_size'], unsupervised=False)
    val_data_gen = DataGenerator(val_pairs, floatx=np.float32, batch_size=params_train['batch'], frame_size=params_data['frame_size'], hop_size=params_data['hop_size'], unsupervised=False, train=False)

    hist = train_model(model, train_data_gen, val_data_gen)
    
    save_model(model, out_path, 'supervised_model')
    return model

out_path = '/import/c4dm-04/cv/models/guitar_distortion_1_2/'
unsupervised_model, nlfx, train_pairs, val_pairs = train_unsupervised(instrument=GUITAR, fx=DISTORTION, param_setting=fx_param_ids[0], dropout=True, out_path=out_path)

# out_path = '/import/c4dm-04/cv/models/guitar_distortion_1_2/'

# dropout = True
# nlfx = NonLinearFXModel(params_data=params_data, params_train=params_train, dropout=True, dropout_rate=0.3)
# model = nlfx.build()
# opt = Adam(lr=params_train['lr'])
# model.compile(optimizer=opt, loss='mae', metrics=['mae'])
# model.load_weights(os.path.join(out_path, 'unsupervised_model.h5'))

# train_pairs, val_pairs = load_data(instrument=GUITAR, fx=DISTORTION, param_setting=fx_param_ids[0], out_path=out_path)

supervised_model = train_supervised(unsupervised_model, nlfx, train_pairs, val_pairs)