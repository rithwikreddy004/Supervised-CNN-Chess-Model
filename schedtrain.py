import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import ChessCNN
from data_preprocess import build_universal_move_mapping
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Load delta tensors
train_delta = torch.load("train_delta.pt")
val_delta = torch.load("val_delta.pt")

# Compute max absolute value for scaling
max_abs = max(train_delta.abs().max(), val_delta.abs().max())

if max_abs == 0:
    max_abs = 1.0

# Scale deltas to [-1, 1]
train_delta = train_delta / max_abs
val_delta = val_delta / max_abs

torch.save(train_delta, "train_delta.pt")
torch.save(val_delta, "val_delta.pt")

print(f"Delta eval scaled by {max_abs}")

# ----------------------------
# Config
# ----------------------------
TRAIN_X_FILE = "train_X.pt"
TRAIN_MOVES_FILE = "train_moves.pt"
TRAIN_DELTA_FILE = "train_delta.pt"

VAL_X_FILE = "val_X.pt"
VAL_MOVES_FILE = "val_moves.pt"
VAL_DELTA_FILE = "val_delta.pt"

BATCH_SIZE = 128
EPOCHS = 2
LR = 1e-3
VALUE_LOSS_WEIGHT = 1.0

RESUME_CHECKPOINT = None

print("Building move mapping from dataset...")
MOVE_TO_IDX, IDX_TO_MOVE = build_universal_move_mapping()
output_size = len(MOVE_TO_IDX)
print(f"Total move classes: {output_size}")

train_X = torch.load(TRAIN_X_FILE)
train_moves = torch.load(TRAIN_MOVES_FILE)
train_delta = torch.load(TRAIN_DELTA_FILE)

val_X = torch.load(VAL_X_FILE)
val_moves = torch.load(VAL_MOVES_FILE)
val_delta = torch.load(VAL_DELTA_FILE)

train_X = train_X.view(-1, 13, 8, 8)
val_X = val_X.view(-1, 13, 8, 8)

train_dataset = TensorDataset(train_X, train_moves, train_delta)
val_dataset = TensorDataset(val_X, val_moves, val_delta)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessCNN(move_output_size=output_size, dropout=0.2).to(device)

policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# NEW: Learning rate scheduler
'''
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',      # Reduce when validation loss stops decreasing
    factor=0.5,      # Halve LR
    patience=2,      # Wait 1 epoch before reducing
    min_lr=1e-5

)'''

if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
    model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=device))
    print(f"[INFO] Loaded checkpoint from {RESUME_CHECKPOINT}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_policy_loss, total_value_loss = 0.0, 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch", ncols=100)
    for x_batch, move_batch, delta_batch in progress_bar:
        x_batch = x_batch.to(device)
        move_batch = move_batch.to(device)
        delta_batch = delta_batch.to(device)

        optimizer.zero_grad()

        policy_logits, value_pred = model(x_batch)

        policy_loss = policy_loss_fn(policy_logits, move_batch)
        value_loss = value_loss_fn(value_pred.squeeze(), delta_batch)

        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item() * x_batch.size(0)
        total_value_loss += value_loss.item() * x_batch.size(0)

        progress_bar.set_postfix(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item()
        )

    avg_policy_loss = total_policy_loss / len(train_dataset)
    avg_value_loss = total_value_loss / len(train_dataset)

    model.eval()
    val_policy_loss, val_value_loss, correct = 0.0, 0.0, 0
    with torch.no_grad():
        for x_batch, move_batch, delta_batch in val_loader:
            x_batch = x_batch.to(device)
            move_batch = move_batch.to(device)
            delta_batch = delta_batch.to(device)

            policy_logits, value_pred = model(x_batch)

            val_policy_loss += policy_loss_fn(policy_logits, move_batch).item() * x_batch.size(0)
            val_value_loss += value_loss_fn(value_pred.squeeze(), delta_batch).item() * x_batch.size(0)

            preds = policy_logits.argmax(dim=1)
            correct += (preds == move_batch).sum().item()

    avg_val_policy_loss = val_policy_loss / len(val_dataset)
    avg_val_value_loss = val_value_loss / len(val_dataset)
    avg_val_loss = avg_val_policy_loss + VALUE_LOSS_WEIGHT * avg_val_value_loss  # NEW: Combined average loss

    val_acc = correct / len(val_dataset) * 100

    print(f"\nEpoch {epoch}/{EPOCHS} | "
          f"Train Policy Loss: {avg_policy_loss:.4f}, Train Value Loss: {avg_value_loss:.4f} | "
          f"Val Policy Loss: {avg_val_policy_loss:.4f}, Val Value Loss: {avg_val_value_loss:.4f}, "
          f"Val Acc: {val_acc:.2f}%")

    # Step the scheduler using combined validation loss
    #scheduler.step(avg_val_loss)

torch.save(model.state_dict(), "chess_cnn.pt")
print("Model saved as chess_cnn.pt")
