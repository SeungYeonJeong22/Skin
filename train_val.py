import torch
import time
import copy
import os
from utils.utils import get_lr, loss_epoch, plot_activation
import wandb


# function to start training
def train_val(model, params, train_loader, val_loader, device='cpu', activation=None, layer_name=None):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    lr_scheduler = params['lr_scheduler']
    model_save_root_path = params['model_save_root_path']
    model_name = params['model_name']
    early_stopping_patience = params['early_stopping_patience']

    model_save_path = os.path.join(model_save_root_path, model_name)

    loss_history = {'train': [], 'val': []}
    acc_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    os.makedirs('train_val_log', exist_ok=True)
    with open(f'./train_val_log/{model_name}.txt', 'w') as f:
        f.write("Training Parameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        for epoch in range(1, num_epochs+1):
            start_time = time.time()
            current_lr = get_lr(opt)
            log_str = 'Epoch {}/{}, current lr= {}\n'.format(epoch, num_epochs, current_lr)
            print(log_str, end='')
            f.write(log_str)

            model.train()
            train_loss, train_acc = loss_epoch(model, model_name, epoch, loss_func, train_loader, opt, device, mode='train')
            loss_history['train'].append(train_loss)
            acc_history['train'].append(train_acc)

            model.eval()
            with torch.no_grad():
                val_loss, val_acc = loss_epoch(model, model_name, epoch, loss_func, val_loader, device=device, mode='valid')
            loss_history['val'].append(val_loss)
            acc_history['val'].append(val_acc)

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr
            })            

            if val_loss < best_loss and val_acc > best_acc:
                best_acc = max(acc_history['val'])

                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)
                log_str = 'Save best model weights!\n'
                print(log_str, end='')
                f.write(log_str)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                log_str = 'Loading best model weights!\n'
                print(log_str, end='')
                f.write(log_str)
                model.load_state_dict(best_model_wts)

            log_str = 'train loss: {:.6f}, val loss: {:.6f}, accuracy: {:.2f}, time: {:.4f} min\n'.format(
                train_loss, val_loss, 100*val_acc, (time.time()-start_time)/60)
            print(log_str, end='')
            f.write(log_str)
            log_str = '-'*10 + '\n'
            print(log_str, end='')
            f.write(log_str)

            if epochs_no_improve >= early_stopping_patience:
                log_str = 'Early stopping triggered. No improvement for {} epochs.\n'.format(early_stopping_patience)
                print(log_str, end='')
                f.write(log_str)
                break

            # 특정 에포크마다 feature 시각화
            if epoch % 5 == 0:
                plot_activation(activation, layer_name, model_name, epoch)

    model.load_state_dict(best_model_wts)
    return model, loss_history, acc_history