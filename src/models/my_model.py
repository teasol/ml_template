import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    """
    Educational simple MLP model.
    Only contains the architecture, independent of the training system.
    """
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        
        # 계층 정의 (Layers)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Input: (Batch, Channels, Height, Width) or (Batch, Input_Dim)
        Output: (Batch, Num_Classes)
        """
        # 만약 이미지가 들어온다면 평탄화(Flatten)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x