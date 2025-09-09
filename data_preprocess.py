'''
import torch
import numpy as np
import random
import chess  # for constants

# ----------------------------
# Config
# ----------------------------
INPUT_FILE = r'C:\honey\OneDrive\Desktop\out1rd.txt'   # your TXT file path
TRAIN_SPLIT = 0.8  # fraction of games for training

# Output files
OUTPUT_TRAIN_X = "train_X.pt"
OUTPUT_TRAIN_MOVES = "train_moves.pt"
OUTPUT_TRAIN_DELTA = "train_delta.pt"

OUTPUT_VAL_X = "val_X.pt"
OUTPUT_VAL_MOVES = "val_moves.pt"
OUTPUT_VAL_DELTA = "val_delta.pt"

# ----------------------------
#
# ----------------------------
PIECE_TO_INDEX = {
    'P': 0,  'N': 1,  'B': 2,  'R': 3,  'Q': 4,  'K': 5,
    'p': 6,  'n': 7,  'b': 8,  'r': 9,  'q': 10, 'k': 11
}
NUM_PIECES = 12

# ----------------------------
# Build universal move mapping (fixed space)
# ----------------------------
def build_universal_move_mapping():
    move_set = set()
    files = "abcdefgh"
    ranks = "12345678"

    # Generate all normal moves (no promotions)
    for from_file in files:
        for from_rank in ranks:
            from_sq = from_file + from_rank
            for to_file in files:
                for to_rank in ranks:
                    to_sq = to_file + to_rank
                    if from_sq != to_sq:
                        move_set.add(from_sq + to_sq)

    # Add promotions
    promotion_pieces = ['q', 'r', 'b', 'n']
    for from_file in files:
        for promo in promotion_pieces:
            move_set.add(from_file + '7' + from_file + '8' + promo)  # white
            move_set.add(from_file + '2' + from_file + '1' + promo)  # black

    move_to_idx = {move: idx for idx, move in enumerate(sorted(move_set))}
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}

    print(f"Total universal moves: {len(move_to_idx)}")
    return move_to_idx, idx_to_move

# ----------------------------
# FEN → tensor (13, 8, 8) for CNN
# ----------------------------
def fen_to_tensor(fen):
    parts = fen.split()
    board_str = parts[0]
    turn = parts[1]  # 'w' or 'b'

    # 12 channels for pieces + 1 channel for side to move
    tensor = np.zeros((NUM_PIECES + 1, 8, 8), dtype=np.float32)

    squares = board_str.split('/')
    for rank_idx, rank in enumerate(squares):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                piece_idx = PIECE_TO_INDEX[char]
                tensor[piece_idx, rank_idx, file_idx] = 1.0
                file_idx += 1

    # Add the extra channel: all 1s if white to move, all 0s if black
    tensor[NUM_PIECES, :, :] = 1.0 if turn == 'w' else 0.0

    return tensor


# ----------------------------
# Group positions by games (now with delta_eval)
# ----------------------------
def group_positions_by_game(file_path):
    START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    games = []
    current_game = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                continue
            fen, uci_move, delta_eval = parts

            # Convert delta_eval to float safely
            try:
                delta_eval = float(delta_eval)
            except ValueError:
                continue

            # Detect start of a new game
            if fen.split()[0] == START_FEN and current_game:
                games.append(current_game)
                current_game = []

            current_game.append((fen, uci_move, delta_eval))

    if current_game:
        games.append(current_game)

    return games

# ----------------------------
# Convert games to tensors
# ----------------------------
def games_to_tensors(games, move_to_idx):
    X_list, move_list, delta_list = [], [], []
    for game in games:
        for fen, uci_move, delta_eval in game:
            x = fen_to_tensor(fen)
            X_list.append(x)

            move_idx = move_to_idx.get(uci_move, 0)  # fallback to 0
            move_list.append(move_idx)

            delta_list.append(delta_eval)

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    move_tensor = torch.tensor(np.array(move_list), dtype=torch.long)
    delta_tensor = torch.tensor(np.array(delta_list), dtype=torch.float32)

    return X_tensor, move_tensor, delta_tensor

# ----------------------------
# Main
# ----------------------------
def main():
    print("Building universal move mapping...")
    MOVE_TO_IDX, IDX_TO_MOVE = build_universal_move_mapping()
    print(f"Universal move space size: {len(MOVE_TO_IDX)}")

    print("Grouping positions by game...")
    games = group_positions_by_game(INPUT_FILE)
    print(f"Total games found: {len(games)}")

    random.shuffle(games)
    split_idx = int(len(games) * TRAIN_SPLIT)
    train_games = games[:split_idx]
    val_games = games[split_idx:]

    print(f"Train games: {len(train_games)}, Validation games: {len(val_games)}")

    print("Converting training games to tensors...")
    train_X, train_moves, train_delta = games_to_tensors(train_games, MOVE_TO_IDX)
    print("Converting validation games to tensors...")
    val_X, val_moves, val_delta = games_to_tensors(val_games, MOVE_TO_IDX)

    # Save
    torch.save(train_X, OUTPUT_TRAIN_X)
    torch.save(train_moves, OUTPUT_TRAIN_MOVES)
    torch.save(train_delta, OUTPUT_TRAIN_DELTA)

    torch.save(val_X, OUTPUT_VAL_X)
    torch.save(val_moves, OUTPUT_VAL_MOVES)
    torch.save(val_delta, OUTPUT_VAL_DELTA)

    print(f"Saved train tensors: {len(train_X)} positions, validation: {len(val_X)} positions")
    print(f"Train_X shape: {train_X.shape}, Train_moves: {train_moves.shape}, Train_delta: {train_delta.shape}")

if __name__ == "__main__":
    main()
'''


import torch
import numpy as np
import random
import os
import chess  # for constants

# ----------------------------
# Config
# ----------------------------
INPUT_FILE = r'C:\Users\honey\OneDrive\Desktop\out13rd.txt'   # your TXT file path
TRAIN_SPLIT = 0.8  # fraction of positions for training


# Output files
OUTPUT_TRAIN_X =     r'C:\Users\honey\OneDrive\Desktop\chunks\out13c\train_X.pt'
OUTPUT_TRAIN_MOVES = r'C:\Users\honey\OneDrive\Desktop\chunks\out13c\train_moves.pt'
OUTPUT_TRAIN_DELTA = r'C:\Users\honey\OneDrive\Desktop\chunks\out13c\train_delta.pt'

OUTPUT_VAL_X =     r'C:\Users\honey\OneDrive\Desktop\chunks\out13c\val_X.pt'
OUTPUT_VAL_MOVES = r'C:\Users\honey\OneDrive\Desktop\chunks\out13c\val_moves.pt'
OUTPUT_VAL_DELTA = r'C:\Users\honey\OneDrive\Desktop\chunks\out13c\val_delta.pt'

# ----------------------------
PIECE_TO_INDEX = {
    'P': 0,  'N': 1,  'B': 2,  'R': 3,  'Q': 4,  'K': 5,
    'p': 6,  'n': 7,  'b': 8,  'r': 9,  'q': 10, 'k': 11
}
NUM_PIECES = 12

# ----------------------------
# Build universal move mapping (fixed space)
# ----------------------------
def build_universal_move_mapping():
    move_set = set()
    files = "abcdefgh"
    ranks = "12345678"

    # Generate all normal moves (no promotions)
    for from_file in files:
        for from_rank in ranks:
            from_sq = from_file + from_rank
            for to_file in files:
                for to_rank in ranks:
                    to_sq = to_file + to_rank
                    if from_sq != to_sq:
                        move_set.add(from_sq + to_sq)

    # Add promotions
    promotion_pieces = ['q', 'r', 'b', 'n']
    for from_file in files:
        for promo in promotion_pieces:
            move_set.add(from_file + '7' + from_file + '8' + promo)  # white
            move_set.add(from_file + '2' + from_file + '1' + promo)  # black

    move_to_idx = {move: idx for idx, move in enumerate(sorted(move_set))}
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}

    print(f"Total universal moves: {len(move_to_idx)}")
    return move_to_idx, idx_to_move

# ----------------------------
# FEN → tensor (13, 8, 8) for CNN
# ----------------------------
def fen_to_tensor(fen):
    parts = fen.split()
    board_str = parts[0]
    turn = parts[1]  # 'w' or 'b'

    # 12 channels for pieces + 1 channel for side to move
    tensor = np.zeros((NUM_PIECES + 1, 8, 8), dtype=np.float32)

    squares = board_str.split('/')
    for rank_idx, rank in enumerate(squares):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                piece_idx = PIECE_TO_INDEX[char]
                tensor[piece_idx, rank_idx, file_idx] = 1.0
                file_idx += 1

    # Add the extra channel: all 1s if white to move, all 0s if black
    tensor[NUM_PIECES, :, :] = 1.0 if turn == 'w' else 0.0

    return tensor

# ----------------------------
# Read positions directly (no grouping by game)
# ----------------------------
def read_positions(file_path):
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                continue
            fen, uci_move, delta_eval = parts

            try:
                delta_eval = float(delta_eval)
            except ValueError:
                continue

            positions.append((fen, uci_move, delta_eval))

    return positions

# ----------------------------
# Convert positions to tensors
# ----------------------------
def positions_to_tensors(positions, move_to_idx):
    X_list, move_list, delta_list = [], [], []
    for fen, uci_move, delta_eval in positions:
        x = fen_to_tensor(fen)
        X_list.append(x)

        move_idx = move_to_idx.get(uci_move, 0)  # fallback to 0
        move_list.append(move_idx)
        delta_list.append(delta_eval)

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    move_tensor = torch.tensor(np.array(move_list), dtype=torch.long)
    delta_tensor = torch.tensor(np.array(delta_list), dtype=torch.float32)

    return X_tensor, move_tensor, delta_tensor

# ----------------------------
# Main
# ----------------------------
def main():
    print("Building universal move mapping...")
    MOVE_TO_IDX, IDX_TO_MOVE = build_universal_move_mapping()
    print(f"Universal move space size: {len(MOVE_TO_IDX)}")

    print("Reading positions directly (no game grouping)...")
    positions = read_positions(INPUT_FILE)
    print(f"Total positions found: {len(positions)}")

    random.shuffle(positions)
    split_idx = int(len(positions) * TRAIN_SPLIT)
    train_positions = positions[:split_idx]
    val_positions = positions[split_idx:]

    print(f"Train positions: {len(train_positions)}, Validation positions: {len(val_positions)}")

    print("Converting training positions to tensors...")
    train_X, train_moves, train_delta = positions_to_tensors(train_positions, MOVE_TO_IDX)
    print("Converting validation positions to tensors...")
    val_X, val_moves, val_delta = positions_to_tensors(val_positions, MOVE_TO_IDX)

    # Save
    torch.save(train_X, OUTPUT_TRAIN_X)
    torch.save(train_moves, OUTPUT_TRAIN_MOVES)
    torch.save(train_delta, OUTPUT_TRAIN_DELTA)

    torch.save(val_X, OUTPUT_VAL_X)
    torch.save(val_moves, OUTPUT_VAL_MOVES)
    torch.save(val_delta, OUTPUT_VAL_DELTA)

    print(f"Saved train tensors: {len(train_X)} positions, validation: {len(val_X)} positions")
    print(f"Train_X shape: {train_X.shape}, Train_moves: {train_moves.shape}, Train_delta: {train_delta.shape}")

if __name__ == "__main__":
    main()
