import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from functions.transformation import augmentation_dataset, get_transform


def load_data():
    
    data = pd.read_csv('..//..//fer2013.csv') #mettre le fichier a la source du projet
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []

    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype('float32'))
    
    face_test = faces
    
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    # Normalize the pixels
    faces /= 255.0

    # Emotion labels
    emotions = pd.get_dummies(data['emotion']).values

    # Convert to PyTorch tensors
    X = torch.tensor(faces, dtype=torch.float32)
    y = torch.tensor(emotions, dtype=torch.long)
    
    #use train test split to split our data into 80% training 20% testing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    
    
    #augmentation_dataset do not accept pytorch tensor as input
    transform = get_transform()
    #transform the data to the train dataset but not the validation one
    faces_data, X_validation = train_test_split(face_test, test_size=0.2, random_state=1)
    dataset = augmentation_dataset(faces_data, transform)
    
    return dataset, X_val, y_train, y_val



def split_data (X_val, y_val, batch_size = 32) :

    # Créer des objets DataLoader pour les ensembles de validation
    test_dataset = data_utils.TensorDataset(X_val, y_val)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader
