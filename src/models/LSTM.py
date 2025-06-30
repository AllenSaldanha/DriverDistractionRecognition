import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KeypointLSTM(nn.Module):
    """
    LSTM-based model for driver activity recognition using pose keypoints
    """
    def __init__(self, num_classes, max_persons=5, hidden_size=128, num_layers=2, dropout=0.3):
        super(KeypointLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.max_persons = max_persons
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input dimension: max_persons * 17 keypoints * 2 coordinates = max_persons * 34
        self.input_size = max_persons * 17 * 2
        
        # Feature embedding for keypoint data
        self.keypoint_embed = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, max_persons, 17, 2]
        batch_size, seq_len = x.shape[:2]
        
        # Flatten keypoints: [batch_size, sequence_length, max_persons * 17 * 2]
        x = x.view(batch_size, seq_len, -1)
        
        # Embed keypoints
        x = self.keypoint_embed(x)  # [batch_size, sequence_length, 128]
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, sequence_length, hidden_size * 2]
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size * 2]
        
        # Classification
        output = self.classifier(last_output)  # [batch_size, num_classes]
        
        return output

class KeypointGRU(nn.Module):
    """
    GRU-based model - often more efficient than LSTM
    """
    def __init__(self, num_classes, max_persons=5, hidden_size=128, num_layers=2, dropout=0.3):
        super(KeypointGRU, self).__init__()
        
        self.num_classes = num_classes
        self.max_persons = max_persons
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_size = max_persons * 17 * 2
        
        # Feature embedding
        self.keypoint_embed = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, max_persons, 17, 2]
        batch_size, seq_len = x.shape[:2]
        
        # Flatten keypoints
        x = x.view(batch_size, seq_len, -1)
        
        # Embed keypoints
        x = self.keypoint_embed(x)
        
        # GRU processing
        gru_out, hidden = self.gru(x)
        
        # Use the last output
        last_output = gru_out[:, -1, :]
        
        # Classification
        output = self.classifier(last_output)
        
        return output

class KeypointTransformer(nn.Module):
    """
    Transformer-based model for keypoint sequences
    Often performs better than RNNs for sequence modeling
    """
    def __init__(self, num_classes, max_persons=5, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(KeypointTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.max_persons = max_persons
        self.d_model = d_model
        
        self.input_size = max_persons * 17 * 2
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, max_persons, 17, 2]
        batch_size, seq_len = x.shape[:2]
        
        # Flatten keypoints
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, input_size]
        
        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer processing
        x = self.transformer(x)  # [batch_size, seq_len, d_model]
        
        # Global average pooling or use last token
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Classification
        output = self.classifier(x)
        
        return output

class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class KeypointCNN1D(nn.Module):
    """
    1D CNN for keypoint sequences - alternative to RNNs
    Often more efficient and can capture local temporal patterns well
    """
    def __init__(self, num_classes, max_persons=5, dropout=0.3):
        super(KeypointCNN1D, self).__init__()
        
        self.num_classes = num_classes
        self.max_persons = max_persons
        self.input_size = max_persons * 17 * 2
        
        # 1D Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(self.input_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Third conv block
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, max_persons, 17, 2]
        batch_size, seq_len = x.shape[:2]
        
        # Flatten keypoints and transpose for conv1d
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, input_size]
        x = x.transpose(1, 2)  # [batch_size, input_size, seq_len]
        
        # Convolutional processing
        x = self.conv_layers(x)  # [batch_size, 512, reduced_seq_len]
        
        # Global pooling
        x = self.global_pool(x)  # [batch_size, 512, 1]
        x = x.squeeze(-1)  # [batch_size, 512]
        
        # Classification
        output = self.classifier(x)
        
        return output

class KeypointAttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for better temporal modeling
    """
    def __init__(self, num_classes, max_persons=5, hidden_size=128, num_layers=2, dropout=0.3):
        super(KeypointAttentionLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.max_persons = max_persons
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_size = max_persons * 17 * 2
        
        # Feature embedding
        self.keypoint_embed = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, max_persons, 17, 2]
        batch_size, seq_len = x.shape[:2]
        
        # Flatten keypoints
        x = x.view(batch_size, seq_len, -1)
        
        # Embed keypoints
        x = self.keypoint_embed(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch_size, sequence_length, hidden_size * 2]
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # [batch_size, sequence_length, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_size * 2]
        
        # Classification
        output = self.classifier(context)
        
        return output