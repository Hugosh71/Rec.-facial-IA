import torch
import torch.nn as nn

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
    def __init__(self,  num_classes=7, in_channels = 1, lr = 0.01,  dropout = 0.5, num_hidden = 4096, model_name = "ResNet"):
        super(ResNet9, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lr = lr
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.model_name = model_name
        self.conv1 = conv_block(in_channels, 16, pool=False) # 16 x 48 x 48
        self.conv2 = conv_block(16, 32, pool=True) # 32 x 24 x 24
        self.res1 = nn.Sequential( #  32 x 24 x 24
            conv_block(32, 32, pool=False), 
            conv_block(32, 32, pool=False)
        )

        self.conv3 = conv_block(32, 64, pool=True) # 64 x 12 x 12
        
        self.res2 = nn.Sequential( # 128 x 6 x 6
             conv_block(64, 64), 
             conv_block(64, 64)
        )
        
        self.conv4 = conv_block(64, 128, pool=True) # 128 x 6 x 6

        self.res3 = nn.Sequential( # 128 x 6 x 6
             conv_block(128, 128), 
             conv_block(128, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # 128 x 3 x 3
            nn.Flatten(),
            nn.Linear(128*3*3, self.num_hidden), #512
            nn.Linear(self.num_hidden, num_classes) # 7
        )
        self.network = nn.Sequential(
            self.conv1,
            self.conv2,
            self.res1,
            self.conv3,
            self.res2,
            self.conv4,
            self.res3,
            self.classifier,
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.res2(out) + out
        out = self.conv4(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out   