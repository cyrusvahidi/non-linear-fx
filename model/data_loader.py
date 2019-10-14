import torch
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
) 

from librosa.core import load

from os import listdir
from os.path import join

class DataLoader(DataLoaderBase):
  def __init__(self, dataset, batch_size, seq_len, overlap_len, *args, **kwargs):
    super().__init__(dataset, batch_size, *args, **kwargs)
    self.seq_len = seq_len
    self.overlap_len = overlap_len

  def __iter__(self):
    for batch in super().__iter__():
      (batch_size, n_samples) = batch.size()

      reset = True

      for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
        from_index 

  def __len__(self):
    raise NotImplementedError()
