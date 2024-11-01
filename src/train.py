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

# Function to plot the loss curve
def plot_loss_curve(losses_train, losses_val = None):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(losses_train, label='Train')
    if losses_val:
        plt.plot(losses_val, label='Validation')
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
        epoch_loss = train_one_epoch(model, optimizer, train_loader, epoch,  CONFIG, criterion)
        losses_train.append(epoch_loss)
        #Evaluate the model on validation set
        if valid_loader:
            epoch_val_loss = validate_one_epoch(model, valid_loader, epoch, CONFIG, criterion)
            losses_val.append(epoch_val_loss)
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
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    return model, losses_train, losses_val, lr_history