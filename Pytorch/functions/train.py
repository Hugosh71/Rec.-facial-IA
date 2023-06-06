from functions.Architectures.ResNet import ResNet9
from functions.Architectures.Model48 import Model48
from functions.Architectures.OurModel import EmotionRecognitionModel
from functions.Architectures.VGG import VGG16
from functions.save import save_model, graphics_loss_acc
from torch import nn, optim
from torch.utils import data as data_utils
from tqdm import tqdm
import torch
import os

def fit(model, train_dataset, test_loader, y_train, batch_size, date, optimizer, criterion, epochs=10):

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
        
        X_train = train_dataset.__getitem__()
        train_tensor = data_utils.TensorDataset(X_train, y_train)
        train_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        
        for inputs, targets in train_loader :
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
    
    save_model(date, train_losses, train_accuracies, test_losses, test_accuracies, model, optimizer, epochs)
    graphics_loss_acc( train_losses, train_accuracies, test_losses, test_accuracies, model.model_name, date)
        
    return train_losses, train_accuracies, test_losses, test_accuracies

def train_model(train_dataset, test_loader, y_train, date, model_name = "Model48", lr=0.001, batch_size=32, dropout = 0.2, num_hidden = 4096,  epochs=10, save = False):
    if model_name == "ResNet" :
        model = ResNet9(num_classes=7, in_channels=1 ,lr=lr, dropout=dropout, num_hidden=num_hidden)
    elif model_name == "Model48" :
        model = Model48(num_classes=7, in_channels=1 ,lr=lr, dropout=dropout, num_hidden=num_hidden)
    elif model_name == "OurModel":
        model = EmotionRecognitionModel(num_classes=7, in_channels=1 ,lr=lr, dropout=dropout, num_hidden=num_hidden)
    elif model_name == "VGG16":
        model = VGG16(num_classes=7, in_channels=1 ,lr=lr, dropout=dropout, num_hidden=num_hidden)
    else :
        raise NameError("Model name not found")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)

    return fit(model, train_dataset, test_loader, y_train, batch_size, date, optimizer, criterion, epochs=epochs)