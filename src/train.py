import os
import time
import datetime

from meta import *
from utils import *
from model.DataGenerator import DataGenerator
from model.NonLinearFXModel import NonLinearFXModel

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

def train(instrument, fx, param_setting, dropout):
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
    sess = tf.Session(config=config) 
    K.set_session(sess)

    input_target_file_pairs = get_input_target_fname_pairs(instrument, fx, param_setting)

    ## PUT AUDIO Loading into a function - do it in data_loader?
    n_total_clips = len(input_target_file_pairs)
    input_target_pairs =  [0] * n_total_clips
    for idx, (f_input, f_target) in enumerate(input_target_file_pairs):
        audio_in = load_audio(os.path.join(params_path[instrument][NO_FX], f_input), idx, n_total_clips, NO_FX)
        audio_target = load_audio(os.path.join(params_path[instrument][fx], f_target), idx, n_total_clips, fx)

        input_target_pairs[idx] = (audio_in, audio_target)

    train_pairs, val_pairs = train_test_split(input_target_pairs,
                                           test_size=params_train.get('val_split'),
                                           random_state=42)


    # Generate input target frame pairs
    train_data_gen = DataGenerator(train_pairs, floatx=np.float32, batch_size=params_train['batch'], frame_size=params_data['frame_size'], hop_size=params_data['hop_size'], unsupervised=True)
    val_data_gen = DataGenerator(val_pairs, floatx=np.float32, batch_size=params_train['batch'], frame_size=params_data['frame_size'], hop_size=params_data['hop_size'], unsupervised=True, train=False)

    start = time.time()

    now = datetime.datetime.now()
    print("Current date and time:")
    print(str(now))

    tr_loss, val_loss = [0] * params_train.get('n_epochs'), [0] * params_train.get('n_epochs')

    model = NonLinearFXModel(params_data=params_data, params_train=params_train, dropout=dropout, dropout_rate=0.3).build()
    opt = Adam(lr=params_train['lr'])
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    model.summary()

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

    path = '/import/c4dm-04/cv/models/unsupervised1/'

    model.save_weights(os.path.join(path, 'model.h5'))

    # Save the model architecture
    with open(os.path.join(path, 'model.json'), 'w') as f:
        f.write(model.to_json())

train(instrument=GUITAR, fx=DISTORTION, param_setting=fx_param_ids[0], dropout=True)