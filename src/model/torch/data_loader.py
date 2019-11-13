import torch
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
) 

import numpy as np

from os import listdir
from os.path import join

class DataLoader(DataLoaderBase):
    def __init__(self, dataset, batch_size, frame_sz, hop_sz, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.frame_sz = frame_sz
        self.hop_sz = hop_sz

    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()

            reset = True

            for seq_begin in range(self.overlap_len, n_samples, self.frame_sz):
                from_index = seq_begin - self.overlap_len
                to_index = seq_begin + self.seq_len
                sequences = batch[:, from_index : to_index]
                input_sequences = sequences[:, : -1]
                target_sequences = sequences[:, self.overlap_len :].contiguous()

                yield (input_sequences, reset, target_sequences)

                reset = False

    def __len__(self):
        raise NotImplementedError()

class DataGenerator(Dataset):
    def __init__(self, dataset, batch_size, frame_size, hop_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.frame_size = frame_size
        
        self.get_input_target_frames()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.input_frames[idx], self.target_frames[idx]
        
    def __len__(self):
        return self.n_frames_total
    
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
        self.nb_iterations = int(np.floor(self.n_frames_total / self.batch_size))
        
        self.input_frames = np.zeros((self.n_frames_total, self.frame_size, 1), dtype=self.floatx)
        self.target_frames = np.zeros((self.n_frames_total, self.frame_size, 1), dtype=self.floatx)
        
        for idx in range(self.n_clips):
            self.get_clip_to_frames(idx)
        
    def get_clip_to_frames(self, idx):
        """ slice the specified clip index into frames
            according to frame length and hop size.
            store the input and target frames
        """
        # indexes that the frames will be stored between for given clips in self.input_frames and self.target_frames
        idx_start = self.nb_inst_cum[idx]   # start for current clip
        idx_end = self.nb_inst_cum[idx + 1] # end for current clip
        
        input_clip = self.dataset[idx][0]
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
        return np.maximum(1, int(np.ceil(n_samples - params_data['frame_size']) / params_data['hop_size']))
        
