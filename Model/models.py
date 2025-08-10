import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

class FD_CNNGRU(nn.Module):
    def __init__(self, n_feat=90, n_complex=2, n_classes=2):
        super().__init__()
        self.n_input = n_feat * n_complex
        self.conv1 = nn.Conv1d(self.n_input, 128, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.transpose(1, 2)  # [batch, 180, 2000]
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.transpose(1, 2)  # [batch, time, features]
        out, _ = self.gru(x)   # [batch, seq, 128]
        out = out[:, -1, :]    # last time step
        x = self.fc(out)
        return x
class FD_1DCNN(nn.Module):
    def __init__(self, seq_len=2000, n_feat=90, n_complex=2, n_classes=2):
        super().__init__()
        self.n_input = n_feat * n_complex  # 180

        self.conv1 = nn.Conv1d(self.n_input, 128, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=9, padding=4)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveMaxPool1d(1)  # global pooling

        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: [batch, 2000, 90, 2] → [batch, 2000, 180]
        x = x.view(x.shape[0], x.shape[1], -1)
        # [batch, 2000, 180] → [batch, 180, 2000]
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = nn.functional.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(self.bn3(x))
        x = self.pool3(x)   # [batch, 256, 1]
        x = x.squeeze(-1)   # [batch, 256]
        x = self.fc(x)      # [batch, 2]
        return x
        
class Gait_MLP(nn.Module):
    def __init__(self, inputsize, num_classes, hidden_sizes=[1024, 256]):
        super().__init__()
        self.fc1 = nn.Linear(inputsize, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        # Add dropout if needed
        # self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerE(nn.Module):
    def __init__(self, input_dim=180, d_model=256, nhead=4, num_layers=2, num_classes=11, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(90*2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, 90, 2]
        b, seq_len, n, c = x.shape
        x = x.view(b, seq_len, n * c)           # [b, seq, 180]
        x = self.input_proj(x)                  # [b, seq, d_model]
        x = x.transpose(0, 1)                   # [seq, b, d_model] for Transformer
        out = self.transformer(x)               # [seq, b, d_model]
        out = out.mean(dim=0)                   # [b, d_model] - average over sequence
        out = self.classifier(out)              # [b, num_classes]
        return out