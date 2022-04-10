import time
import torch
import numpy as np
import random
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from datasets.dataset import SIRSimulatedData
from models.rnn import Network

from sklearn.metrics import accuracy_score
import gc
import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

BATCH_SIZE = 64
# LEARNING_RATE = 1e-3
LEARNING_RATE = 1e-6
# LEARNING_RATE = 0.00035
EPOCHS = 5

def train(model, train_loader, optimizer, criterion, scaler, scheduler, epoch):

    model.train()

    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False)
    total_loss = 0

    for batch_idx, (X, y) in enumerate(train_loader):

        optimizer.zero_grad()

        X, y = X.float().cuda(), y.float().cuda()

        with autocast():
            output = model(X)
            loss = criterion(output, y)
        
        total_loss += float(loss)
        batch_bar.set_postfix(
            loss=f"{float(total_loss / (batch_idx + 1)):.4f}",
            lr=f"{float(optimizer.param_groups[0]['lr']):.6f}"
        )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_bar.update()
        
        # torch.cuda.empty_cache()
    
    scheduler.step()

    batch_bar.close()
    print(f"Epoch {epoch}/{EPOCHS}: Train Loss {float(total_loss / len(train_loader)):.04f}, Learning Rate {float(optimizer.param_groups[0]['lr']):.06f}")

def validation(model, val_loader, criterion, epoch, val_mse, save_model=True):

    model.eval()

    epoch_mse = []

    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False)

    for batch_idx, (X, y) in enumerate(val_loader):

        X, y = X.float().cuda(), y.long().cuda()

        with torch.no_grad():
            output = model(X)
        
        mse = criterion(output, y)
        
        batch_bar.set_postfix(MSE=f"{float(mse):.4f}")
        batch_bar.update()
        epoch_mse.append(mse)

    batch_bar.close()
    epoch_mse = np.array(epoch_mse)
    print(f"Validation: Batch-averged MSE: {epoch_mse.mean():.4f}")
    val_mse.append(epoch_mse.mean())

    if save_model:
        model_name = 'model_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        model_path = f'model/trained_models/{model_name}_checkpoint_{epoch}.pkl'
        torch.save(model, model_path)
    

def main():
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    epochs = EPOCHS

    train_data = SIRSimulatedData(partition='train')
    val_data = SIRSimulatedData(partition='dev')
    # test_data = SIRSimulatedData(partition='test')

    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn,
    #                           num_workers=16, pin_memory=True) # TODO: Define the train loader. Remember to pass in a parameter (function) for the collate_fn argument 
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True) # TODO: Define the val loader. Remember to pass in a parameter (function) for the collate_fn argument 
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True) # TODO: Define the test loader. Remember to pass in a parameter (function) for the collate_fn argument 

    model = Network(4, 256, bidirectional=True)
   
    # CHECKPOINT_MODEL_DIR = "model/trained_models/model_20220408_111048_checkpoint_5.pkl"
    # model = torch.load(CHECKPOINT_MODEL_DIR).cuda()
    

    criterion = nn.MSELoss() # TODO: What loss do you need for sequence to sequence models? 
    # Do you need to transpose or permute the model output to find out the loss? Read its documentation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5) # TODO: Adam works well with LSTM (use lr = 2e-3)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-5, nesterov=True)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    torch.cuda.empty_cache()

    model.cuda()
    scaler = GradScaler()
    val_mse= []
    # train_loader = list(train_loader)
    # val_loader = list(val_loader)
    for epoch in range(1, epochs + 1):

        train(model, train_loader, optimizer, criterion, scaler, scheduler, epoch)

        if (epoch % 5 == 0) & (epoch != 0):
            validation(model, val_loader, criterion, epoch, val_mse)

    # CHECKPOINT_MODEL_DIR = "model/trained_models/model_20220405_133933_checkpoint_40.pkl"
    # model = torch.load(CHECKPOINT_MODEL_DIR).cuda()


    log_df = pd.DataFrame({"epoch": list(range(5, EPOCHS + 1, 5)),
                           "validation_lev_distance": val_mse})
    log_name = 'log_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_df.to_csv(f"log/{log_name}.csv")


    # TODO: Write the model evaluation function if you want to validate after every epoch

    # You are free to write your own code for model evaluation or you can use the code from previous homeworks' starter notebooks
    # However, you will have to make modifications because of the following.
    # (1) The dataloader returns 4 items unlike 2 for hw2p2
    # (2) The model forward returns 2 outputs
    # (3) The loss may require transpose or permuting

    # Note that when you give a higher beam width, decoding will take a longer time to get executed
    # Therefore, it is recommended that you calculate only the val dataset's Levenshtein distance (train not recommended) with a small beam width
    # When you are evaluating on your test set, you may have a higher beam width

if __name__ == '__main__':
    main()