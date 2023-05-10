import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from datetime import datetime
from os import path
import os

# Load the data
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

# Define the hyperparameter grid to search
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [32, 64],
    'dropout': [0.25, 0.5],
    'epochs': [50, 100, 200]
}

# Initialize the grid search
grid = ParameterGrid(param_grid)

# Initialize lists to store performance metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

#Get the date
d = datetime.now()
date = d.strftime('%m-%d')
        
#Crate necessary folder
folder_name = "modele"
if not os.path.exists(folder_name): os.makedirs(folder_name)

folder_name = f"modele/cross_val_{date}"
if not os.path.exists(folder_name): os.makedirs(folder_name)

folder_name = "graph"
if not os.path.exists(folder_name): os.makedirs(folder_name)

folder_name = f"graph/cross_val_{date}"
if not os.path.exists(folder_name): os.makedirs(folder_name)


# Define the model class
class EmotionRecognitionModel(nn.Module):
    def __init__(self, dropout, lr=0.01, batch_size=64):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 7)
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.epochs = 100

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
    
def cross_val_score(model, X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    scores = []
    for fold, (train_indices, val_indices) in enumerate(kfold.split(X)):
        # Separate the data into training and validation sets
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        val_dataset = TensorDataset(X[val_indices], y[val_indices])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model.batch_size)

        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model.lr)
        epochs = model.epochs
        train_losses = []
        train_precision = []
        train_recall = []
        train_f1 = []
        
        for epoch in tqdm(range(epochs), desc="Traitement en cours", bar_format="{l_bar}{bar:10}{r_bar}"):
            model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            for inputs, targets in train_loader:
                inputs = inputs.permute(0, 3, 1, 2)
                optimizer.zero_grad()
                outputs = model(inputs[:targets.size(0)])
                loss = criterion(outputs, torch.max(targets, 1)[1])
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += targets.size(0)
                correct_train += predicted.eq(torch.max(targets, 1)[1]).sum().item()

                true_positives += ((predicted == 1) & (targets[:,1] == 1)).sum().item()
                false_positives += ((predicted == 1) & (targets[:,1] == 0)).sum().item()
                false_negatives += ((predicted == 0) & (targets[:,1] == 1)).sum().item()

            # Compute performance metrics for the training set
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            train_accuracy = correct_train / len(train_loader.dataset)
            train_precision.append(true_positives / (true_positives + false_positives + 1e-8))
            train_recall.append(true_positives / (true_positives + false_negatives + 1e-8))
            train_f1.append(2 * train_precision[-1] * train_recall[-1] / (train_precision[-1] + train_recall[-1] + 1e-8))

            # Evaluate the model on the validation set
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.permute(0, 3, 1, 2)
                    outputs = model(inputs[:targets.size(0)])
                    loss = criterion(outputs, torch.max(targets, 1)[1])

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_val += targets.size(0)
                    correct_val += predicted.eq(torch.max(targets, 1)[1]).sum().item()
                    true_positives += ((predicted == 1) & (targets[:,1] == 1)).sum().item()
                    false_positives += ((predicted == 1) & (targets[:,1] == 0)).sum().item()
                    false_negatives += ((predicted == 0) & (targets[:,1] == 1)).sum().item()

                # Compute performance metrics for the validation set
                val_loss /= len(val_loader)
                val_accuracy = correct_val / len(val_loader.dataset)
                val_precision = true_positives / (true_positives + false_positives + 1e-8)
                val_recall = true_positives / (true_positives + false_negatives + 1e-8)
                val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-8)

                print(f'Fold {fold + 1}/{n_splits}, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train Precision: {train_precision[-1]:.4f}, Train Recall: {train_recall[-1]:.4f}, Train F1: {train_f1[-1]:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

            metrics_dict = {
                'lr': model.lr,
                'batch_size': model.batch_size,
                'dropout': model.dropout1.p,
                'train_loss': np.mean(train_losses),
                'train_accuracy': train_accuracy,
                'train_precision': np.mean(train_precision),
                'train_recall': np.mean(train_recall),
                'train_f1': np.mean(train_f1),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            }
            scores.append(metrics_dict)

        # Compute mean and standard deviation of the scores
        mean_scores = {key: np.mean([x[key] for x in scores]) for key in scores[0]}
        std_scores = {key: np.std([x[key] for x in scores]) for key in scores[0]}
        mean_scores.update(std_scores)
        print(mean_scores)
        
        
        architecture = ""
        nom_fichier = f"fold{fold+1}_tr-acc{(train_accuracy*100):.1f}_val-acc{(val_accuracy*100):.1f}"
        file_path = f'modele/cross_val_{date}/{nom_fichier}.pth'
        
        torch.save({
            'epochs': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': model.lr,
            'batch_size': model.batch_size,
            'dropout': model.dropout,
            'architecture': architecture,
            'train_loss': np.mean(train_losses),
            'train_accuracy': train_accuracy,
            'train_precision': np.mean(train_precision),
            'train_recall': np.mean(train_recall),
            'train_f1': np.mean(train_f1),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'mean_score' : mean_scores
        }, file_path)
        
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Test Loss')
        plt.legend()
        plt.title(f"K_Fold : {fold+1}")
        plt.savefig(f'graph/cross_val_{date}/loss_Kfold{fold+1}.png')

        plt.figure()
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Test Accuracy')
        plt.legend()
        plt.title(f"K_Fold : {fold+1}")
        plt.savefig(f'graph/cross_val_{date}/acc_Kfold{fold+1}.png')
        
        
model = EmotionRecognitionModel(dropout=0.25, lr=0.001, batch_size=16)
scores = cross_val_score(model, X, y, n_splits=5)
print(scores)
