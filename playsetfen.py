import torch
import numpy as np
import chess
from model import ChessCNN  # CNN with policy + value heads
import pickle

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "ultrachess_cnn1.pt"
NUM_PIECES = 12
BOARD_SHAPE = (13, 8, 8)  # 12 pieces + 1 side-to-move channel
print(MODEL_PATH+" is loading...")
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

    # Side to move channel
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
        for move in board.legal_moves:
            move_uci = move.uci()
            if move_uci in MOVE_TO_IDX:
                mask[0, MOVE_TO_IDX[move_uci]] = 1.0

        masked_probs = policy_probs * mask
        if masked_probs.sum() == 0:
            # fallback: pick first legal move
            best_move = list(board.legal_moves)[0].uci()
            top5 = [(best_move, 1.0)]
        else:
            masked_probs /= masked_probs.sum()
            top5_idx = masked_probs[0].topk(5).indices.tolist()
            top5 = [(IDX_TO_MOVE[idx], masked_probs[0, idx].item()) for idx in top5_idx]
            best_move = IDX_TO_MOVE[top5_idx[0]]

        return best_move, top5, value_pred.item()

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
# FEN testing mode
# ----------------------------
# ----------------------------
# FEN testing mode
# ----------------------------
def fen_testing_loop():
    while True:
        fen_input = input("\nEnter FEN (or 'q' to quit): ").strip()
        if fen_input.lower() in ["q", "quit", "exit"]:
            print("Exiting FEN testing mode.")
            break

        try:
            board = chess.Board(fen_input)
        except ValueError:
            print("Invalid FEN! Try again.")
            continue

        print("\nCurrent board:")
        print(board)

        ai_uci, top5, value_pred = predict_move(board.fen(), board)

        print(f"\nPredicted evaluation: {value_pred:.6f}")
        print("Top 5 predicted moves:")
        for move, prob in top5:
            print(f"{uci_to_san(board, move)} → {prob:.3f}")

        print(f"\nBest move: {ai_uci} → {uci_to_san(board, ai_uci)}")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":

    fen_testing_loop()

