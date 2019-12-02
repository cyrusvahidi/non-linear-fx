from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence

import numpy as np

from os import listdir
from os.path import join

class DataGenerator(Sequence):
    def __init__(self, dataset, floatx, batch_size, frame_size, hop_size, unsupervised=False, train=True):
        self.dataset      = dataset
        self.floatx       = floatx
        self.batch_size   = batch_size
        self.frame_size   = frame_size
        self.hop_size     = hop_size
        self.unsupervised = unsupervised
        self.train = train
        
        self.get_input_target_frames()
        
        self.on_epoch_end()
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
 
        return self.input_frames[indexes], self.target_frames[indexes]
    
    def __len__(self):
#         return self.n_iterations
        return self.n_iterations
    
    def on_epoch_end(self):
        # shuffle data between epochs
        self.indexes = np.random.permutation(self.n_frames_total)
        
    def get_input_target_frames(self):
        # load the audio
        # count the frames
        print("Loading audio frames")
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_frames_per_clip(input_clip)
                   for input_clip, _ in self.dataset], dtype=int))
        
        self.n_clips = len(self.dataset)
        
        self.n_frames_total = self.nb_inst_cum[-1]
        
        # how many batches can we fit in the set
        if self.train:
            self.n_iterations = 1000
        else:
            self.n_iterations = int(np.floor(self.n_frames_total / self.batch_size))
        
        self.input_frames = np.zeros((self.n_frames_total, self.frame_size, 1), dtype=self.floatx)
        self.target_frames = np.zeros((self.n_frames_total, self.frame_size, 1), dtype=self.floatx)
        
        for idx in range(self.n_clips):
            self.get_clip_to_frames(idx)
        
        print('Loaded {0} frames'.format(self.n_frames_total))
        
    def get_clip_to_frames(self, idx):
        """ slice the specified clip index into frames
            according to frame length and hop size.
            store the input and target frames
        """
        # indexes that the frames will be stored between for given clips in self.input_frames and self.target_frames
        idx_start = self.nb_inst_cum[idx]   # start for current clip
        idx_end = self.nb_inst_cum[idx + 1] # end for current clip
        
        input_clip = self.dataset[idx][0]
        if self.unsupervised:
            target_clip = input_clip
        else:
            target_clip = self.dataset[idx][1]
        
        idx = 0    # index to start placing frames in self.input_frames and self.target_frames
        start = 0  # start frame within input and target clips
        
        while idx < (idx_end - idx_start):
            self.input_frames[idx_start + idx] = input_clip[start:start + self.frame_size]
            self.target_frames[idx_start + idx] = target_clip[start:start + self.frame_size]
            
            start += self.hop_size
            idx += 1
                              
        
    def get_num_frames_per_clip(self, audio_clip):
        # get the number of frames for given clip
        n_samples = audio_clip.shape[0]
        return np.maximum(1, int(np.ceil(n_samples - self.frame_size) / self.hop_size))
        
