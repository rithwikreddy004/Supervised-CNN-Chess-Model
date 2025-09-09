# Supervised CNN Chess Model

A Python-based supervised learning chess engine that predicts optimal moves from board positions using a Convolutional Neural Network (CNN). This project demonstrates AI-driven move prediction, optimized network architecture, and data-driven inference.

## Project Overview

Traditional chess engines use Minimax search and heuristics to evaluate moves. This project explores a **data-driven approach using supervised learning**:

- Trains a CNN on millions of chess positions (FENs) from high-quality games.
- Predicts optimal moves in SAN (Standard Algebraic Notation) format.
- Mimics both human and engine play styles.
- Focused on fast inference rather than exhaustive search, enabling quicker move suggestions.

## Key Features

- **Move Prediction**: Given a board state, predicts the most likely next move.  
- **Supervised Learning**: Trains on a large dataset of game positions and moves.  
- **Optimized CNN Architecture**: Designed to handle board representation efficiently.  
- **Data-Driven**: Learns from real game patterns rather than predefined rules.  
- **Extensible**: Can be further trained or integrated into GUI-based chess applications.

## Technologies Used

- **Python**: Programming and scripting  
- **PyTorch**: Deep learning framework for building the CNN  
- **NumPy**: Data manipulation  
- **Pandas**: Dataset preprocessing  
- **Chess Python Library**: Parsing and handling FENs

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/supervised-cnn-chess.git

