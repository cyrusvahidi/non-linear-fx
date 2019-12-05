import yaml
import pandas as pd
import os

params = yaml.safe_load(open("config.yaml"))

params_data = params['data']
params_train = params['train']

suffix = params_data['suffix']

path_root = params['data_path']

BASS = 'bass'
GUITAR = 'guitar'

NO_FX = 'nofx'
DISTORTION = 'distortion'
OVERDRIVE = 'overdrive'

fx_param_ids = ['1', '2', '3'] # identifiers for the parameter settings

params_path = {
    BASS: {
        NO_FX: os.path.join(path_root, 'bass/NoFX/'), 
        DISTORTION: os.path.join(path_root, 'bass/Distortion/'), 
        OVERDRIVE: os.path.join(path_root, 'bass/Overdrive/')
    },
    GUITAR: {
        NO_FX: os.path.join(path_root, 'guitar/NoFX/'),
        DISTORTION: os.path.join(path_root, 'guitar/Distortion/'), 
        OVERDRIVE: os.path.join(path_root, 'guitar/Overdrive/')
    }
}

path_models = '/import/c4dm-04/cv/models/'
