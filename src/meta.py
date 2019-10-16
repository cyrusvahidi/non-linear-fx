import yaml
import pandas as pd
import os

params = yaml.safe_load(open("config.yaml"))

params_data = params['data']
params_train = params['train']

suffix = params_data['suffix']

path_root = params['data_path']

params_path = {
    'bass_nofx': os.path.join(path_root, 'bass/NoFX/'),
    'bass_distortion': os.path.join(path_root, 'bass/Distortion/'),
    'bass_overdrive': os.path.join(path_root, 'bass/Overdrive/'),
    'guitar_nofx': os.path.join(path_root, 'guitar/NoFX/'),
    'guitar_distortion': os.path.join(path_root, 'guitar/Distortion/'),
    'guitar_overdrive': os.path.join(path_root, 'guitar/Overdrive/')
}