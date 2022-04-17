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

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20

def train(model, train_loader, optimizer, criterion, scaler, scheduler, epoch):

    model.train()

    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False)
    total_loss = 0

    for batch_idx, (X, y) in enumerate(train_loader):

        optimizer.zero_grad()

        X, y = X.float().cuda(), y.float().cuda()

        # with autocast():
        output = model(X).squeeze()
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
            output = model(X).squeeze()
        
        mse = criterion(output, y)
        
        batch_bar.set_postfix(MSE=f"{float(mse):.4f}")
        batch_bar.update()
        epoch_mse.append(float(mse.cpu()))

    batch_bar.close()
    epoch_mse = np.array(epoch_mse)
    print(f"Validation: Batch-averged MSE: {epoch_mse.mean():.4f}")
    val_mse.append(epoch_mse.mean())

    if save_model:
        model_name = 'model_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        model_path = f'models/trained_models/{model_name}_checkpoint_{epoch}.pkl'
        torch.save(model, model_path)


def main():
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    epochs = EPOCHS

    train_data = SIRSimulatedData(partition='train')
    val_data = SIRSimulatedData(partition='dev')
    # test_data = SIRSimulatedData(partition='test')
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True) 

    model = Network(4, 256, bidirectional=True)
   
    # CHECKPOINT_MODEL_DIR = "model/trained_models/model_20220408_111048_checkpoint_5.pkl"
    # model = torch.load(CHECKPOINT_MODEL_DIR).cuda()
    
    criterion = nn.L1Loss() 
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

        if (epoch % 2 == 0) & (epoch != 0):
            validation(model, val_loader, criterion, epoch, val_mse)

    # CHECKPOINT_MODEL_DIR = "model/trained_models/model_20220405_133933_checkpoint_40.pkl"
    # model = torch.load(CHECKPOINT_MODEL_DIR).cuda()


    log_df = pd.DataFrame({"epoch": list(range(5, EPOCHS + 1, 5)),
                           "validation_lev_distance": val_mse})
    log_name = 'log_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_df.to_csv(f"log/{log_name}.csv")

if __name__ == '__main__':
    main()