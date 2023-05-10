import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from os import path
import os



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
# Normalisation des pixels
faces /= 255.0
# Étiquettes d'émotion
emotions = pd.get_dummies(data['emotion']).values
# Conversion en tenseurs PyTorch
X = torch.tensor(faces, dtype=torch.float32)
y = torch.tensor(emotions, dtype=torch.long)
# Création d'un ensemble de données
dataset = TensorDataset(X, y)
# Séparation des données en ensembles d'entraînement et de test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
generator1 = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
# Création des chargeurs de données (DataLoaders)
batch_size = 32
lr = 0.001
dropout= 0.25
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Création des dossier nécéssaire pour le code
folder_name = "modele"
if not os.path.exists(folder_name): os.makedirs(folder_name)

folder_name = "modele/cross_val"
if not os.path.exists(folder_name): os.makedirs(folder_name)


print('DATA LOADED')

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

model = EmotionRecognitionModel(dropout, lr, batch_size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=model.lr)


# Initialisation des listes pour stocker les métriques de performance
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

epochs = 50

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

    #print(f'Epoch: {epoch}, Train Loss: {train_loss / total}, Train Acc: {correct / total}, Test Loss: {test_loss / total_test}, Test Acc: {correct_test / total_test}')


# Enregistrement des poids du modèle et de l'optimiseur à la fin du run
d = datetime.now()
date = d.strftime('%m-%d')
architecture = ""
nom_fichier = f"{date}_tr-acc{(train_accuracy*100):.1f}_te-acc{(test_accuracy*100):.1f}"
file_path = "modele/"+nom_fichier+'.pth'
torch.save({
    'epochs': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'train_accuracy': train_accuracy,
    'test_loss': test_loss,
    'test_accuracy': test_accuracy,
    'lr' : lr,
    'batch_size': batch_size,
    'dropout' : dropout,
    'architecture' : architecture
}, file_path)

if os.path.exists(file_path):
    print("File saved successfully!")
else:
    print("Error: File not saved.")
    

# Tracer les graphiques d'évolution de la précision et de la perte pour chaque ensemble de données
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend()
plt.show()




