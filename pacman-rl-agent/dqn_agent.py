# DQN Agent implementation
import torch
import torch.nn as nn

class DQNAgent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 512), nn.ReLU(),
            nn.Linear(512, num_actions)      
        )
    
    def forward(self, x):
        return self.net(x)
