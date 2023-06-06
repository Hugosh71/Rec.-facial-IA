import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionRecognitionModel(nn.Module):
    def __init__(self,  num_classes=7, in_channels = 1, lr = 0.01,  dropout = 0.5, num_hidden = 4096, model_name = "EmotionRecognitionModel"):
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
        self.model_name = model_name


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