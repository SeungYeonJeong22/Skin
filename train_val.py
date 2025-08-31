import torch
import time
import copy
import os
from utils.utils import get_lr, loss_epoch
import wandb
import pandas as pd


# function to start training
def train_val(model, params, train_loader, val_loader, int2label, device='cpu', activation=None, layer_name=None, wandb_flag=False, training_record_flag='false'):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    lr_scheduler = params['lr_scheduler']
    model_save_root_path = params['model_save_root_path']
    model_name = params['model_name']
    save_model_path = params['save_model_path']
    early_stopping_patience = params['early_stopping_patience']
    tag = params['tag']

    train_results_save_path = ""

    model_save_path = os.path.join(model_save_root_path, save_model_path)
    if training_record_flag == 'true':
        training_record_flag = True
        os.makedirs("./train_records", exist_ok=True)
        train_results_save_path = os.path.join("./train_records", save_model_path.split(".")[0] + '.csv')
    else:
        training_record_flag = False

    loss_history = {'train': [], 'val': []}
    acc_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}\n'.format(epoch, num_epochs, current_lr))

        model.train()

        train_loss, train_acc, train_records = loss_epoch(model, model_name, epoch, loss_func, train_loader, int2label=int2label, opt=opt, device=device, mode='train', training_record_flag=training_record_flag)
        loss_history['train'].append(train_loss)
        acc_history['train'].append(train_acc)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, model_name, epoch, loss_func, val_loader, int2label=int2label, device=device, mode='valid', training_record_flag=training_record_flag)
        loss_history['val'].append(val_loss)
        acc_history['val'].append(val_acc)

        if wandb_flag:
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
            # best_model = torch.save(model, model_save_path)
               
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_path)
            print('Save best model weights!\n')
            epochs_no_improve = 0

            if training_record_flag:
                pd.DataFrame(train_records).to_csv(train_results_save_path, index=False)
                # 갱신된 train_records를 저장
                print('Save train records to {}'.format(train_results_save_path))

        else:
            print('Early stopping patience: {} / {}'.format(epochs_no_improve, early_stopping_patience))
            epochs_no_improve += 1

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print('Loading best model weights!\n')
            model.load_state_dict(best_model_wts)
            # model = torch.load(model_save_path, weights_only=False, map_location=device)

        print('train loss: {:.6f}, val loss: {:.6f}, accuracy: {:.2f}, time: {:.4f} min\n'.format(
            train_loss, val_loss, 100*val_acc, (time.time()-start_time)/60))
        print('-'*10 + '\n')

        if epochs_no_improve >= early_stopping_patience:
            print('Early stopping triggered. No improvement for {} epochs.\n'.format(early_stopping_patience))
            break


    model.load_state_dict(best_model_wts)
    return model, loss_history, acc_history