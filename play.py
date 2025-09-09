import torch
import numpy as np
import chess
from model import ChessCNN  # CNN with policy + value heads
import pickle

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "r1chess_cnn2.pt"
NUM_PIECES = 12
BOARD_SHAPE = (13, 8, 8)  # 12 pieces + 1 side-to-move channel

# ----------------------------
# Load move mapping
# ----------------------------
with open("move_mapping.pkl", "rb") as f:
    MOVE_TO_IDX, IDX_TO_MOVE = pickle.load(f)

OUTPUT_SIZE = len(MOVE_TO_IDX)

# ----------------------------
# FEN → tensor (13, 8, 8)
# ----------------------------
PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

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
# Load model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessCNN(move_output_size=OUTPUT_SIZE, dropout=0.2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Predict move (policy + value)
# ----------------------------
def predict_move(fen, board):
    x = fen_to_tensor(fen)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        policy_logits, value_pred = model(x_tensor)
        policy_probs = torch.softmax(policy_logits, dim=-1)

        # Mask illegal moves
        mask = torch.zeros_like(policy_probs)
        legal_moves = [m.uci() for m in board.legal_moves]
        for move in legal_moves:
            if move in MOVE_TO_IDX:
                mask[0, MOVE_TO_IDX[move]] = 1.0

        masked_probs = policy_probs * mask
        if masked_probs.sum() == 0:
            best_move = legal_moves[0]
            top5 = [(best_move, 1.0)]
        else:
            masked_probs /= masked_probs.sum()
            top5_idx = masked_probs[0].topk(5).indices.tolist()
            top5 = [(IDX_TO_MOVE[idx], masked_probs[0, idx].item()) for idx in top5_idx]
            best_idx = top5_idx[0]
            best_move = IDX_TO_MOVE[best_idx]

        return best_move, top5, value_pred.item()  # value head output included

# ----------------------------
# SAN ↔ UCI
# ----------------------------
def san_to_uci(board, san_move):
    try:
        return board.parse_san(san_move).uci()
    except:
        return None

def uci_to_san(board, uci_move):
    try:
        return board.san(chess.Move.from_uci(uci_move))
    except:
        return None

# ----------------------------
# Interactive play
# ----------------------------
def play_san_input():
    board = chess.Board()
    while not board.is_game_over():
        print("\nCurrent board:")
        print(board)

        # Human turn
        if board.turn == chess.WHITE:
            human_san = input("Your move (SAN, e.g., e4, Nf3): ")
            uci_move = san_to_uci(board, human_san)
            if uci_move is None or chess.Move.from_uci(uci_move) not in board.legal_moves:
                print("Invalid SAN move. Try again.")
                continue
            board.push(chess.Move.from_uci(uci_move))

        # AI turn
        ai_uci, top5, value_pred = predict_move(board.fen(), board)
        print(f"Predicted evaluation (delta): {value_pred:.6f}")
        print("Top 5 predicted moves with probabilities:")
        for move, prob in top5:
            print(f"{uci_to_san(board, move)} → {prob:.3f}")

        if chess.Move.from_uci(ai_uci) in board.legal_moves:
            ai_san = uci_to_san(board, ai_uci)
            print(f"AI plays: {ai_uci} → {ai_san}")
            board.push(chess.Move.from_uci(ai_uci))
        else:
            print("fallback!")
            ai_move = list(board.legal_moves)[0]
            ai_san = board.san(ai_move)
            print(f"AI fallback move: {ai_move.uci()} → {ai_san}")
            board.push(ai_move)

    print("\nGame over!")
    print(f"Result: {board.result()}")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    play_san_input()

