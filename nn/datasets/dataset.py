import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SIRSimulatedData(Dataset):
    
    def __init__(self, path='data/data.pkl', partition='train'):
        super(SIRSimulatedData).__init__()
    
        with open(path, 'rb') as f:
            
            data, labels = pickle.load(f)
            
        if partition == 'train':
            self.data, self.labels = data[:16000], labels[:16000]
        
        elif partition == 'dev':
            self.data, self.labels = data[16000:18000], labels[16000:18000]
        
        elif partition == 'test':
            self.data, self.labels = data[18000:], labels[18000:]
            
        
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self, index):
        
        return self.data[index], self.labels[index]
        