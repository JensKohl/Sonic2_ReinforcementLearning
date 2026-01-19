# Sonic 2 PPO Reinforcement Learning Agent

This project demonstrates how to build a Reinforcement Learning (RL) agent to play *Sonic the Hedgehog 2* using Proximal Policy Optimization (PPO).

## 🎯 Goal
To provide a clear, educational, and modular implementation of PPO from scratch using PyTorch, explaining key concepts like:
- **PPO (Proximal Policy Optimization)**: The core learning algorithm.
- **CNN (Convolutional Neural Networks)**: For processing game frames.
- **Frame Stacking**: To give the agent a sense of motion.
- **Parallel Environments**: To speed up training.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `uv` package manager
- **Sonic 2 ROM**: You must provide `Sonic the Hedgehog 2 (JUE) [!].bin` in the `ROMS/` directory.

### Installation
1. Initialize the environment:
   ```bash
   uv sync
   ```
2. Import the game into stable-retro:
   ```bash
   python -m retro.import ROMS
   ```

### Usage
- **Train**:
  ```bash
  uv run python src/train.py
  ```
  This will start training using the GPU. Logs are saved to `logs/` and models to `models/`.

- **Evaluate**:
  ```bash
  uv run python -m src.evaluate --model models/your_model_name.pth
  ```
  This will open a window and show the agent playing. Use `--model` to specify a checkpoint. Without it, a random agent plays.


## 📂 Project Structure
- `src/`: Source code for the agent and training.
- `docs/`: Educational documentation on RL concepts.
- `models/`: Checkpoints of trained models.
- `logs/`: Tensorboard logs.

## 📚 Documentation
Check out [docs/reinforcement_learning.md](docs/reinforcement_learning.md) to learn about the theory behind the code!
