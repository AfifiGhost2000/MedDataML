import torch.nn as nn
import torch.nn.functional as F


class ParallelCNN(nn.Module):
    def __init__(self, num_classes=1):  
        super(ParallelCNN, self).__init__()

        # Branch 1 (filter size 1)
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, padding=0), # Assuming input channels = 3 (RGB)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to ensure same output size
        )

        # Branch 2 (filter size 2)
        self.conv_branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to ensure same output size
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2, 512),  # Adjusted input size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Output for your number of classes
        )

    def forward(self, x):
        out1 = self.conv_branch1(x)
        out2 = self.conv_branch2(x)
        out = torch.cat([out1, out2], dim=1)  # Concatenate along the channel dimension
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out


