import yaml
import pandas as pd
import os

params = yaml.safe_load(open("config.yaml"))

params_data = params['data']
params_train = params['train']

suffix = params_data['suffix']

path_root = params['data_path']
path_root_fxchain = '/import/c4dm-04/marcoc/fxdataset/'

BASS = 'bass'
GUITAR = 'guitar'

FXCHAIN = 'fxchain'

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
    },
    FXCHAIN: {
        GUITAR: {
            NO_FX: os.path.join(path_root_fxchain, 'Guitar/NoFX/'),
            '1': os.path.join(path_root_fxchain, 'Guitar/FXChain1/'),
            '2': os.path.join(path_root_fxchain, 'Guitar/FXChain2/'), 
            '3': os.path.join(path_root_fxchain, 'Guitar/FXChain3/')
        },
        BASS: {
            NO_FX: os.path.join(path_root_fxchain, 'Bass/NoFX/'),
            '1': os.path.join(path_root_fxchain, 'Bass/FXChain1/'), 
            '2': os.path.join(path_root_fxchain, 'Bass/FXChain2/'), 
            '3': os.path.join(path_root_fxchain, 'Bass/FXChain3/')
        }
    }
}

path_models = '/import/c4dm-04/cv/models/'
