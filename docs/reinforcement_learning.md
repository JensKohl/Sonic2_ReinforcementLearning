# Understanding Reinforcement Learning for Sonic 2

## What is Reinforcement Learning?
Reinforcement Learning (RL) is an area of machine learning where an **agent** learns to make decisions by performing **actions** in an **environment** and receiving **rewards**.

## Key Concepts

### 1. The Environment (Sonic 2)
The environment gives the agent:
- **Observation**: The current screen pixel data (what Sonic sees).
- **Reward**: Points for moving right, collecting rings, or finishing levels.
- **Done**: A signal if the level is finished or Sonic dies.

### 2. The Agent (PPO)
We use **Proximal Policy Optimization (PPO)**. It's a popular "Actor-Critic" algorithm.
- **Actor**: Decides which button to press based on the screen. It outputs probabilities (e.g., 80% chance to jump).
- **Critic**: Estimates how "good" the current situation is (Value Function). It helps the Actor learn by telling it if a move led to a better-than-expected outcome.

### 3. Convolutional Neural Network (CNN)
Since the input is an image (pixels), we use a CNN to "see".
- It detects edges, shapes, and objects (like Sonic, rings, enemies).
- We use a specific architecture often called "Nature CNN" (from the DeepMind Nature paper).

### 4. Frame Stacking
A single image doesn't show motion. Is Sonic running left or right?
To solve this, we stack 4 consecutive frames together. This gives the agent information about **velocity** and **acceleration**.

### 5. Parallel Training
RL is slow. To speed it up, we run multiple copies of Sonic 2 at the same time (vectorized environments). The agent collects experience from all of them simultaneously.
### 6. Implementation Details

#### Environment Wrappers (`src/env_wrappers.py`)
To make Sonic suitable for a Neural Network, we apply several transformations:
1.  **SonicDiscretizer**: The Genesis controller has 12 buttons. We simplify this to 7 distinct actions (Left, Right, Jump, etc.) to make learning easier.
2.  **ResizeObservation**: Original game is 320x224. We resize to 84x84 squares to reduce computation.
3.  **TransposeObservation**: PyTorch expects images in `(Channels, Height, Width)` format, but Gym provides `(Height, Width, Channels)`. This wrapper fixes that.
4.  **FrameStack**: We stack 4 frames. With RGB color (3 channels), the input to the CNN becomes 12 channels (4 frames x 3 channels).

#### Neural Network (`src/cnn.py`, `src/agent.py`)
-   **Input**: `(Batch, 12, 84, 84)`
-   **Structure**: 3 Convolutional Layers -> Flatten -> 512 Neurons -> Actor/Critic Heads.
-   **Output**: 
    -   Actor: 7 probabilities (one for each action).
    -   Critic: 1 value (how good is the current state).
