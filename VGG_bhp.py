import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
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
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import optuna
from optuna import Trial
import torch.utils.data as data_utils
from random import randint





#    "VGG16": [
#         64,
#         64,
#         "M",
#         128,
#         128,
#         "M",
#         256,
#         256,
#         256,
#         "M",
#         512,
#         512,
#         512,
#         "M",
#         512,
#         512,
#         512,
#         "M",
#     ],

def conv_block(in_channels, out_channels, pool=False):
    layers = [
              nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)
    ]

    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2))
    
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self,  num_classes=7, in_channels = 1, lr = 0.01,  dropout = 0.5, num_hidden = 4096):
        super(ResNet9, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lr = lr
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.conv1 = conv_block(in_channels, 16, pool=False) # 16 x 48 x 48
        self.conv2 = conv_block(16, 32, pool=True) # 32 x 24 x 24
        self.res1 = nn.Sequential( #  32 x 24 x 24
            conv_block(32, 32, pool=False), 
            conv_block(32, 32, pool=False)
        )

        self.conv3 = conv_block(32, 64, pool=True) # 64 x 12 x 12
        self.conv4 = conv_block(64, 128, pool=True) # 128 x 6 x 6

        self.res2 = nn.Sequential( # 128 x 6 x 6
             conv_block(128, 128), 
             conv_block(128, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # 128 x 3 x 3
            nn.Flatten(),
            nn.Linear(128*3*3, 512), #512
            nn.Linear(512, num_classes) # 7
        )
        self.network = nn.Sequential(
            self.conv1,
            self.conv2,
            self.res1,
            self.conv3,
            self.conv4,
            self.res2,
            self.classifier,
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out       
        
        
class Model48(nn.Module):
    def __init__(self,  num_classes=7, in_channels = 1, lr = 0.01,  dropout = 0.5, num_hidden = 4096):
        super(Model48, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lr = lr
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.network = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 16 x 24 x 24

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 12 x 12

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 6 x 6

            nn.Flatten(), 
            nn.Linear(128*6*6, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout-0.1),
            nn.Linear(512, self.num_classes))


    def forward(self, x):
        return self.network(x)
    


class EmotionRecognitionModel(nn.Module):
    def __init__(self,  num_classes=7, in_channels = 1, lr = 0.01,  dropout = 0.5, num_hidden = 4096):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 7)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lr = lr
        self.dropout = dropout
        self.num_hidden = num_hidden


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    

class MyVGG16(nn.Module):
    def __init__(self,  num_classes=7, in_channels = 1, lr = 0.01,  dropout = 0.5, num_hidden = 4096):
        super(MyVGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Conv2d(256, 512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512, self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.num_hidden, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        # x = torch.flatten(x, 1)
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
        
        
        for inputs, targets in train_loader:
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

def train_model(train_loader, test_loader, X_train, X_val, y_train, y_val, lr=0.001, batch_size=32, dropout = 0.2, num_hidden = 4096,  epochs=10, save = False):

    model = Model48(num_classes=7, in_channels=1 ,lr=lr, dropout=dropout, num_hidden=num_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)

    if save == True :
        return model, fit(model, train_loader, test_loader, optimizer, criterion, epochs=epochs)
    else :
        return fit(model, train_loader, test_loader, optimizer, criterion, epochs=epochs)

#call the function above to train our model

X,y = load_data()
#split our data
train_loader, test_loader, X_train, X_val, y_train, y_val = split_data(X,y, batch_size=32)
#traoin our model
train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(train_loader, test_loader, X_train, X_val, y_train, y_val, lr=0.001, batch_size=16, dropout = 0.2, num_hidden = 4096,  epochs=10)


# def objective(trial, X, y):
    
#     lr = trial.suggest_float('lr', 0.0001, 0.1)
#     dropout = trial.suggest_float('dropout', 0.1, 0.7)
#     epochs = trial.suggest_int('epochs', 5, 10)
#     batch_size = trial.suggest_categorical('batch_size', [16, 32])

#     train_loader, test_loader, X_train, X_val, y_train, y_val = split_data(X,y, batch_size=batch_size)
#     # Entraînement du modele avec la fonction train_model
#     train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(train_loader, test_loader, X_train, X_val, y_train, y_val, lr=lr, epochs=epochs)
#     return val_acc_history[-1]
    

# func = lambda trial: objective(trial, X, y)

# study = optuna.create_study(direction = "maximize")
# study.optimize(func, n_trials=4)

# trial = study.best_trial
# #print accuracy and best parameters
# print('Accuracy: {}'.format(trial.value))
# print("Best hyperparameters: {}".format(trial.params))

# #get the best parameters
# lr = trial.params['lr']
# epochs = trial.params['epochs']
# batch_size = trial.params['batch_size']
# dropout = trial.params['dropout']

# #train the model with the best parameters
# train_loader, test_loader, X_train, X_val, y_train, y_val = split_data(X,y)
# model, (train_loss_history_bp, train_acc_history_bp, val_loss_history_bp, val_acc_history_bp) = train_model(train_loader, test_loader, X_train, X_val, y_train, y_val, lr=lr, batch_size=batch_size, dropout=dropout, epochs=epochs, save= True)

# #save the model with pytorch
# torch.save({
#     'epochs': epochs,
#     'model_state_dict': model.state_dict(),
#     'train_loss': train_loss_history_bp,
#     'train_accuracy': train_acc_history_bp,
#     'test_loss': val_loss_history_bp,
#     'test_accuracy': val_acc_history_bp,
#     'lr' : lr,
#     'batch_size': batch_size,
#     'epochs' : epochs,
#     'architecture' : "VGG16"
# }, 'model.pth')
