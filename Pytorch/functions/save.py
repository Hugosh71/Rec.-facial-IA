from os import path
import os
import datetime
import torch
import matplotlib.pyplot as plt

#create folder to save models and graphs
def create_folder(model):
    date = datetime.datetime.now().strftime("%m-%d")
    
    folder_name = "modele"
    if not os.path.exists(folder_name): os.makedirs(folder_name)

    folder_name = f"modele/train_{model}_{date}"
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    
    folder_name = "graphs"
    if not os.path.exists(folder_name): os.makedirs(folder_name)

    folder_name = f"graphs/train_{model}_{date}"
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    
    return date


#save models into the models folder
def save_model(date, train_loss_history, train_acc_history, val_loss_history, val_acc_history, model, optimizer, epochs) :
    nom_fichier = f"_tr-acc{(train_acc_history[-1]*100):.1f}_val-acc{(val_acc_history[-1]*100):.1f}"
    file_path = f'modele/train_{model}_{date}/{nom_fichier}.pth'
    
    torch.save({
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': model.lr,
        'dropout': model.dropout,
        'num_hidden': model.num_hidden,
        'num_classes': model.num_classes,
        'in_channels': model.in_channels,
        'model_name': model.model_name,
        'train_loss': train_loss_history,
        'train_accuracy': train_acc_history,
        'val_loss': val_loss_history,
        'val_accuracy': val_acc_history,
    }, file_path)
    
    
#save graphs into the graphs folder
def graphics_loss_acc(train_loss_history, train_acc_history, val_loss_history, val_acc_history, model_name, date):

        plt.figure()
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Test Loss')
        plt.legend()
        plt.title(f"Graphe de perte pour {model_name}")	
        plt.savefig(f'graphs/train_{model_name}_{date}/loss_{model_name}_acc_{val_acc_history[-1]}.png')

        plt.figure()
        plt.plot(train_acc_history, label='Train Accuracy')
        plt.plot(val_acc_history, label='Test Accuracy')
        plt.legend()
        plt.title(f"Graphe d'accuracy pour {model_name}")
        plt.savefig(f'graphs/train_{model_name}_{date}/acc_{model_name}_acc_{val_acc_history[-1]}.png')