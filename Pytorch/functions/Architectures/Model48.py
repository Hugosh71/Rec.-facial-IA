import torch
import torch.nn as nn

class Model48(nn.Module):
    def __init__(self,  num_classes=7, in_channels = 1, lr = 0.01,  dropout = 0.5, num_hidden = 4096, model_name = "Model48"):
        super(Model48, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lr = lr
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.model_name = model_name
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