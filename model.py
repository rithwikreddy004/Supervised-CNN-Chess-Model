import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessCNN(nn.Module):
    def __init__(self, move_output_size=None, dropout=0.2):
        """
        Multi-task CNN for chess:
          - Input: (13, 8, 8) board tensor (12 piece planes + 1 turn plane)
          - Outputs:
              1) Move logits: [N, move_output_size]
              2) Delta eval: [N, 1] (float, regression)
        """
        super(ChessCNN, self).__init__()

        if move_output_size is None:
            raise ValueError("move_output_size must be specified (number of move classes)")

        # Convolutional backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),  # UPDATED: 13 channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_output_size = 256 * 8 * 8

        # Shared fully connected base
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Head for move prediction (classification)
        self.move_head = nn.Linear(512, move_output_size)

        # Head for delta evaluation (regression)
        self.delta_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.shared_fc(x)

        move_logits = self.move_head(x)        # For CrossEntropyLoss
        delta_eval = self.delta_head(x).squeeze(-1)  # For MSELoss (shape [N])

        return move_logits, delta_eval
