import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchsummary import summary
import gc
import torch.nn as nn
import time
import numpy as np

# Function to plot the loss curve
def plot_loss_curve(losses_train, losses_val = None):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(losses_train, label='Train')
    if losses_val:
        losses_val = np.array(losses_val)
        plt.plot(losses_val[:,0],losses_val[:,1], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def display_trainable_params(model, img_size = (32,32)):
    
    print(f"Generating summary for model with input size {img_size[0]}x{img_size[1]}")
    summary(model, (1, img_size[0], img_size[1]))
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    return sum([p.numel() for p in trainable_params])


def fetch_scheduler(optimizer, CONFIG):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG["scheduler_params"]['T_max'], 
                                                   eta_min=CONFIG["scheduler_params"]['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG["scheduler_params"]['T_0'], 
                                                             eta_min=CONFIG["scheduler_params"]['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def criterion(outputs, targets, CONFIG):
    if CONFIG['model_params']['out_channels'] ==2:
        targets = targets[:,1:,:,:]
    if CONFIG['loss'] == 'MSE' or CONFIG['loss'] == None:
        return nn.MSELoss()(outputs, targets)
    elif CONFIG['loss'] == 'BCE':
        return nn.BCELoss()(outputs, targets)
    elif CONFIG['loss'] == 'L1':
        return nn.L1Loss()(outputs, targets)
    elif CONFIG['loss'] == 'SmoothL1':
        return nn.SmoothL1Loss()(outputs, targets)
    elif CONFIG['loss'] == 'CrossEntropy':
        return nn.CrossEntropyLoss()(outputs, targets)
    elif CONFIG['loss'] == 'BCEWithLogits':
        return nn.BCEWithLogitsLoss()(outputs, targets)
    else:
        raise ValueError("Invalid loss function")
    
def train_one_epoch(model, optimizer, dataloader, epoch, CONFIG, criterion=criterion ):

    model.train()
    batch_test_loss = 0.0
    num_epochs = CONFIG['epochs']
    pbar = tqdm(dataloader,  desc=f"Epoch {epoch+1}/{num_epochs}",leave=False, total=len(dataloader),position = 0)
    for input, target in pbar:
        input, target = input.to(CONFIG['device']), target.to(CONFIG['device'])
        optimizer.zero_grad()

        outputs = model(input)
        loss = criterion(outputs, target, CONFIG)
        
        loss.backward()
        optimizer.step()

        batch_test_loss += loss.item()*input.size(0)
    epoch_test_loss = batch_test_loss / len(dataloader.dataset)

    gc.collect()
    return epoch_test_loss

@torch.inference_mode()
def validate_one_epoch(model, dataloader, epoch, CONFIG, criterion=criterion):
    model.eval()
    batch_val_loss = 0.0
    pbar = tqdm(dataloader, leave=False, total=len(dataloader),position = 0)
    for input, target in pbar:
            input, target = input.to(CONFIG['device']), target.to(CONFIG['device'])
            outputs = model(input)
            loss = criterion(outputs, target, CONFIG)
            batch_val_loss += loss.item()*input.size(0)
    epoch_val_loss = batch_val_loss / len(dataloader.dataset)
    #tqdm.write(f"Validation Loss: {epoch_val_loss:.4f}")
    gc.collect()
    return epoch_val_loss

def train_model(train_loader,valid_loader,model,optimizer,scheduler,CONFIG,criterion=criterion):
    losses_train = []
    losses_val = []
    lr_history = []
    num_epochs = CONFIG['epochs']
    start = time.time()
    for epoch in range(num_epochs):
        if CONFIG['resume_training']:
            if epoch == 0:
                checkpoint = torch.load(CONFIG['resume_checkpoint'])
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                #scheduler.load_state_dict(checkpoint['scheduler'])
                for key in checkpoint['CONFIG']:
                    if key in CONFIG:
                        CONFIG[key] = checkpoint['CONFIG'][key]
                #CONFIG = checkpoint['CONFIG']
                losses_train = checkpoint['train_loss']
                losses_val = checkpoint['valid_loss']
                lr_history = checkpoint['lr_rate']
                
                epoch_n = len(losses_train)
                epoch_val_loss = losses_val[-1][1]
                if scheduler is not None:
                    scheduler = fetch_scheduler(optimizer, CONFIG)
                    for _ in range(epoch):
                        scheduler.step()
                plot_loss_curve(losses_train, losses_val)
                
                print(f"Attempting to resume training from epoch {len(losses_train)}")
            if epoch < epoch_n:
                print(f"Skipping epoch {epoch+1} until resuming epoch {epoch_n+1}")
                continue
        #Train the model
        epoch_loss = train_one_epoch(model, optimizer, train_loader, epoch,  CONFIG, criterion)
        losses_train.append(epoch_loss)
        #Evaluate the model on validation set
        if valid_loader and ((epoch+1)% CONFIG['validate_every_epoch'] == 0 or epoch == 0 ):
            epoch_val_loss = validate_one_epoch(model, valid_loader, epoch, CONFIG, criterion)
            losses_val.append([epoch,epoch_val_loss])
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
            lr_rate = scheduler.get_last_lr()[0]
            lr_history.append(lr_rate)
        else:
            lr_rate = optimizer.param_groups[0]['lr']
        # Plot the loss curve
        if (epoch+1)% CONFIG['display_loss_epoch'] == 0 or epoch == num_epochs-1:
            plot_loss_curve(losses_train, losses_val)
        
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, LR: {lr_rate:.6f}")
        #Save the model in every save_every_epoch to the disk in case of a crash
        if (epoch +1)%CONFIG['save_every_epoch'] == 0 or epoch == num_epochs-1:
            checkpoint = {'model': model,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),#'scheduler': scheduler.state_dict(),
                'CONFIG': CONFIG,'train_loss': losses_train,'valid_loss': losses_val,'lr_rate': lr_rate}
            now = time.localtime()
            torch.save(checkpoint, f"../models/tmp/{CONFIG['model']}_{time.strftime('%Y-%m-%d_%H_%M', now)}_ep{epoch+1}.pth")
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    return model, losses_train, losses_val, lr_history