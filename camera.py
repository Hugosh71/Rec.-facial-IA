import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd 
import cv2
import numpy as np
import matplotlib.pyplot as plt



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

# Définition des étiquettes d'émotion
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Chargement de la webcam
cap = cv2.VideoCapture(0)

#chargement du modele


checkpoint=torch.load('modele//05-08_tr-acc67.3_te-acc54.6_bs32_epo50_do0.25_lr0.001_ar.pth')


#A METTRE FICHIER SUIVANTS
# lr = checkpoint['lr']
# dropout = checkpoint['dropout']
# batch_size = checkpoint['batch_size']
# architecture = checkpoint['architecture']

dropout = 0.25
lr = 0.001
batch_size = 32

model = EmotionRecognitionModel(dropout, lr, batch_size)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

while True:
    # Capture d'une image depuis la webcam
    ret, frame = cap.read()

    # Conversion de l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Prétraitement de l'image pour correspondre à l'entrée du modèle
    face = cv2.resize(gray, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=0)

    # Prédiction de l'émotion
    with torch.no_grad():
        output = model(torch.tensor(face))
        predicted_emotion = emotion_labels[torch.argmax(output).item()]

    # Affichage de l'émotion prédite sur l'image capturée
    cv2.putText(frame, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Emotion Recognition', frame)

    # Quitter la boucle en appuyant sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
