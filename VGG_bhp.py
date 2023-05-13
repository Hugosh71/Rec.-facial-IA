import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from datetime import datetime
from os import path
import os
import sklearn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import optuna
from optuna import Trial
import torch.utils.data as data_utils
from random import randint

class MyVGG16(nn.Module):
    def __init__(self, lr = 0.01,  num_classes=7, in_channels = 1):
        super(MyVGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

def load_data():
    len_of_task = randint(3, 20)  # take some random length of time
    
    data = pd.read_csv('fer2013.csv')
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []

    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    # Normalize the pixels
    faces /= 255.0

    # Emotion labels
    emotions = pd.get_dummies(data['emotion']).values

    # Convert to PyTorch tensors
    X = torch.tensor(faces, dtype=torch.float32)
    y = torch.tensor(emotions, dtype=torch.long)
    return X,y


def split_data (X,y, batch_size = 32) :
    #use train test split to split our data into 80% training 20% testing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Créer des objets DataLoader pour les ensembles d'entraînement et de validation
    train_dataset = data_utils.TensorDataset(X_train, y_train)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = data_utils.TensorDataset(X_val, y_val)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_train, X_val, y_train, y_val

#call the function above to split our data


def fit(model, train_loader, test_loader, optimizer, criterion, epochs=10):

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in tqdm(range(epochs), desc="Traitement en cours", bar_format="{l_bar}{bar:10}{r_bar}"):
        # Entraînement
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        
        for inputs, targets in tqdm(train_loader, desc="Entraînement en cours", bar_format="{l_bar}{bar:10}{r_bar}"):
            inputs = inputs.permute(0, 3, 1, 2)  # Changez l'ordre des dimensions pour correspondre à l'entrée du modèle
            optimizer.zero_grad()
            outputs = model(inputs[:targets.size(0)])
            loss = criterion(outputs, torch.max(targets, 1)[1])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0) 
            correct_train += predicted.eq(torch.max(targets, 1)[1]).sum().item()

        # Calcul des métriques de performance pour l'ensemble d'entraînement
        train_loss /= len(train_loader.dataset)
        train_accuracy = correct_train / len(train_loader.dataset)

        # Évaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0 
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.permute(0, 3, 1, 2)
                outputs = model(inputs[:targets.size(0)])  
                loss = criterion(outputs, torch.max(targets, 1)[1])

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0) 
                correct_test += predicted.eq(torch.max(targets, 1)[1]).sum().item()

        # Calcul des métriques de performance pour l'ensemble de test
        test_loss /= len(test_loader.dataset)
        test_accuracy = correct_test / len(test_loader.dataset)

        # Affichage des métriques de performance
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

        # Stocker les métriques de performance pour chaque itération
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
    return train_losses, train_accuracies, test_losses, test_accuracies

def train_model(train_loader, test_loader, X_train, X_val, y_train, y_val, lr=0.001, epochs=10, save = False):

    model = MyVGG16(lr = lr,  num_classes=7, in_channels = 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)

    if save == True :
        return model, fit(model, train_loader, test_loader, optimizer, criterion, epochs=epochs)
    else :
        return fit(model, train_loader, test_loader, optimizer, criterion, epochs=10)

#call the function above to train our model

X,y = load_data()


def objective(trial, X, y):
    
    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    epochs = trial.suggest_int('epochs', 10, 100)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    train_loader, test_loader, X_train, X_val, y_train, y_val = split_data(X,y)
    # Entraînement du modele avec la fonction train_model
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(train_loader, test_loader, X_train, X_val, y_train, y_val, lr=lr, epochs=epochs)
    return val_acc_history[-1]
    

func = lambda trial: objective(trial, X, y)

study = optuna.create_study(direction = "maximize")
study.optimize(func, n_trials=50)

trial = study.best_trial
#print accuracy and best parameters
print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

#get the best parameters
lr = trial.params['lr']
epochs = trial.params['epochs']
batch_size = trial.params['batch_size']

#train the model with the best parameters
train_loader, test_loader, X_train, X_val, y_train, y_val = split_data(X,y)
model, train_loss_history_bp, train_acc_history_bp, val_loss_history_bp, val_acc_history_bp = train_model(train_loader, test_loader, X_train, X_val, y_train, y_val, lr=lr, epochs=epochs, save= True)

#save the model with pytorch
torch.save({
    'epochs': epochs,
    'model_state_dict': model.state_dict(),
    'train_loss': train_loss_history_bp,
    'train_accuracy': train_acc_history_bp,
    'test_loss': val_loss_history_bp,
    'test_accuracy': val_acc_history_bp,
    'lr' : lr,
    'batch_size': batch_size,
    'epochs' : epochs,
    'architecture' : "VGG16"
}, 'model.pth')
