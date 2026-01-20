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
1.  **FrameSkip (4 Frames)**: **Standard RL best practice.** At 60 FPS, the agent changes its "mind" too frequently to build momentum. By repeating an action for 4 frames, we force "commitment," helping Sonic actually build the speed needed for hills and loops.
2.  **SonicDiscretizer**: Simplifies 12 buttons into a few logical game commands. We added "Jump Forward" and "Spin Dash" macros because complex physics maneuvers are hard for an AI to Discover by accident.
3.  **SonicRewardV3 (Biased Momentum)**: Standard RL rewards distance (Progress). However, Sonic needs to backtrack to build speed for loops. We reward **Absolute Velocity** (speed in any direction) but give **3x more reward** for moving Right. This encourages goal-orientation while literally "paying" the agent to build momentum.
4.  **Stagnation Check**: Prevents "Reward Farming." We detect if the agent is just "wiggling" in place to collect velocity points and restart the level if they don't cover 600px every 30 seconds.
5.  **Resize & Transpose**: Standard image processing to make the data 7x smaller and compatible with PyTorch.
6.  **FrameStack**: Adds "Temporal Awareness," allowing the CNN to see velocity.

#### Neural Network (`src/cnn.py`, `src/agent.py`)
-   **Input**: `(Batch, 12, 84, 84)`
-   **Action Space**: 10 distinct actions (expanded to include diagonal jumps).
-   **Entropy (Curiosity)**: Set to `0.02` to ensure Sonic stays curious about different paths if he gets stuck.

---

## Case Study: The "Wiggling Problem" (Reward Hacking)

During development, we encountered a classic Reinforcement Learning pitfall called **Reward Hacking**.

### 1. The Symptom
After ~7.5 million steps of training, we observed the following metrics:
- **Entropy: 0.0001**: The agent's "curiosity" collapsed to zero. It became 100% deterministic.
- **Value Loss: low**: The critic perfectly predicted the rewards.
- **Behavior**: In the simulation, Sonic stopped moving forward and started "jittering" or wiggling back and forth against a wall.

### 2. The Cause (Why it happened)
Reinforcement Learning agents are like "lazy" lawyers—they will find the easiest mathematical way to get a reward, even if it contradicts the goal.

In our `SonicRewardV4`, we used these parameters:
- **Rightward movement**: `+0.06` per pixel.
- **Leftward movement**: `+0.02` per pixel.

**The Loophole**: By moving 2 pixels Right and then 2 pixels Left, the agent earned:
`(2 * 0.06) + (2 * 0.02) = +0.16` total reward every few frames.
Since this reward was "safe" and infinite, the agent decided that wiggling in a corner was better than risking death by trying to clear a loop.

### 3. The Solution: SonicRewardV5
To fix this, we introduced two "Anti-Farming" mechanics:
1. **Velocity Threshold**: We only give momentum rewards if the speed is **greater than 2 pixels**. This makes "micro-wiggling" worth 0 points.
2. **Strict Stagnation**: We force the episode to restart if Sonic doesn't travel at least **600 pixels** every 30 seconds.

**Lesson for Students**: Always check if your "Dense Rewards" (rewards given every frame) can be exploited by repetitive, non-productive actions!
