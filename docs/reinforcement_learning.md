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

---

## Case Study: Reward Scaling (Gradient Stability)

During the final run, we observed **Entropy Collapse**. The agent's curiosity dropped to zero (0.0000) almost immediately.

### 1. The Symptom
The Metrics showed:
- **Entropy: 0.0000** (Update 20)
- **Value Loss: > 2000**

### 2. The Cause (Reward Magnitude)
Our cumulative rewards (Progress + Momentum) were reaching values of **+10 per step**. In Deep RL, these huge numbers cause "Gradient Explosion". The neural network is so shocked by the large rewards that it immediately crushes the exploration logic and "locks in" to whatever action it was doing.

### 3. The Fix: 0.01 Scaling
We applied a global multiplier of **0.01** to all rewards.
- Instead of a +10 reward, the agent Sees **+0.1**.
- This keeps the "Value Function" in a small, stable range.
- **Result**: The agent remains "curious" for millions of steps, allowing it to actually learn the level instead of panic-committing to one move.

---

## Case Study: Hill Mastery (The "Local Minimum" Problem)

In the final phase of training, we observed Sonic getting stuck in front of vertical hills. He would "jitter" and jump repeatedly but never clear the obstacle.

### 1. The Symptom (Local Minimum)
Sonic found a "safe" way to get small rewards:
- **Small Jumps**: He would jump, get a tiny bit of `max_x` progress, fall back, and repeat.
- **Biased Momentum**: Because running right was worth 3x more than running left, he was afraid to run backward far enough to build the momentum needed for the hill.

### 2. The Solution: SonicRewardV6
We implemented two specific fixes:
1. **Unbiased Momentum**: We made momentum rewards the same for both directions. This taught Sonic that "running away" from the hill to build speed is just as valuable as running toward it.
2. **Jump Penalty**: We added a **-0.1 penalty** for every step spent jumping.
   - This makes "stutter-jumping" at a hill actually *cost* points.
   - It forces the agent to explore the "grounded" solution: **Building Momentum**.

**Lesson for Students**: Sometimes your agent finds a "shortcut" that gives small, consistent rewards but prevents them from finding the "big win." Use penalties to discourage these lazy behaviors!

---

## Case Study: Momentum Farming (The Upside-Down Reward)

In Phase 6, we tried to help Sonic by rewarding him for running fast. We thought: "If he likes running fast, he will run up the hill!"

### 1. The FAILURE
Instead of running up the hill, the agent realized:
> "Wait, I get points for running left and right on the flat ground. The hill is dangerous. I will just run back and forth here forever!"

By giving points for *Unbiased Momentum*, we accidentally created a **Reward Farm**. The agent was getting high scores without ever progressing through the level.

### 2. The Solution: Back to Basics (SonicRewardV7)
We realized that **Progress** is the only thing that matters.
- **Removed Momentum Rewards**: Running fast is now worth 0 points.
- **Removed Jump Penalties**: We don't tell him *how* to move, only *where* to go.
- **Result**: The "Value Function" was starving. The agent realized: "I am dying and getting no points. The ONLY thing that makes the number go up is moving RIGHT." This forced him to finally challenge the hill on his own terms.

**Lesson for Students**: Be careful with "Shaped Rewards." Sometimes, telling the agent *how* to play distracts it from *winning* the game!

---

## Case Study: Context-Aware Rewards (The Physics Teacher)

In Phase 8, we found that "Pure Progress" (V7) was too hard. The agent would get stuck at a wall and just stand there, because it didn't know that running *backwards* was the solution to going forwards.

### 1. The Strategy: "Recovery Mode"
We implemented a **State-Based Reward System**:
1.  **Progress Mode (Default)**: Normal rules. Reward `max_x`.
2.  **Recovery Mode (Trigger)**: If Sonic is stuck at the same spot for **4 seconds**...
    - **ENABLE**: Rewards for speed and backtracking.
    - **DISABLE**: Immediately when he breaks his record `max_x`.

### 2. The Impact
This explicitly teaches the agent a complex sequence:
> "If you are stuck, back up and run fast!"

By making the momentum reward conditional (Context-Aware), we get the benefits of shaped rewards (teaching skills) without the downsides (farming loops).

---

## Case Study: Spring Farming (SonicRewardV9/V10)

Even with "Recovery Mode," we found a new loophole: **Spring Loops**.

### 1. The Symptom
The agent would get stuck near a spring (like the one before the waterfall) and bounce on it repeatedly. 
- While in "Recovery Mode," every bounce provided a small burst of momentum reward.
- The agent effectively "farmed" the spring until the stagnation timer kicked in.

### 2. The Fix (V9/V10)
- **V9 (Limiter)**: We added a 10-second timer to Recovery Mode. After 10 seconds, the bonus shut off.
- **V10 (Penalty)**: We added a specific penalty for jumping while in recovery mode.

### 3. The Result: FAILURE
While V9 and V10 stopped the infinite farming, they weren't enough to solve the level. The agent became "lazy"—it stopped farming, but it also stopped trying to clear the waterfall hill because the rewards for backtracking were now too small relative to the effort.

---

---

## Case Study: The Waterfall Plateau (Spatial Stalemate)

In Phase 10, we encountered our most significant hurdle: a complete stop at **x=2703**.

### 1. The Symptom (The Eternal Loop)
After a full 15-million-step run, the agent's progress showed:
- **Max X reached: 2703** (The base of the first waterfall).
- **Log Static**: The `best.pth` model had not updated for **5 hours**.
- **Behavior**: The agent reached the bridge, jittered at the edge, but refused to climb the platforms needed to advance.

### 2. The Cause (Horizontal Blindness)
Standard RL rewards only care about the X-axis (Right). At the waterfall:
- Moving Forward (Right) requires first moving **UP** (Vertical).
- Since moving Up gives 0 points, and the platforming is difficult, the agent decides that standing still is the "safest" way to avoid death while maintaining its high score.
- **Reward Farming**: Even with stagnation checks, the agent found small "wiggle" movements that satisfied the velocity requirements without clearing the hill.

### 3. The Solution: SonicRewardV11 (The Mountain Climber)

To break the stalemate, we introduced **2D Progress** and **Spatial Memory**:

1.  **Altitude Reward (Y-Axis)**: We now reward Sonic for reducing his `y` coordinate (climbing higher). This makes vertical platforming just as profitable as running right.
2.  **Tile-Based Curiosity**: We divide the level into **16x16 pixel tiles**. The first time Sonic enters a new tile, he gets a **Discovery Bonus**. 
    - This creates a "scavenger hunt" effect. 
    - The agent is now encouraged to explore every corner of the screen, even if those corners don't immediately move him further right.

**Lesson for Students**: When your agent hits a complex 3D obstacle (like a hill or a platforming puzzle), horizontal rewards aren't enough. You must reward the **discovery of space** itself!

---

## Case Study: Physics Commitment (SonicRewardV12)

While V11 broke the "Waterfall Plateau," the agent still struggled with the **Long Vertical Ramp** (around x=4200). 

### 1. The Symptom
The agent would reach the base of the ramp, try to run up, lose speed halfway, and slide back down. It never learned the **sequence** of preparation required to clear it.

### 2. The Solution: SonicRewardV12 (The Speedster)

We introduced **Procedural Rewards** targeting those specific physics maneuvers:

1.  **Spin Dash Incentive**: Sonic now gets a small reward for holding `DOWN + B` while stopped. This encourages him to "rev up" his speed before moving.
2.  **Backtrack Credit**: If Sonic is in front of a ramp and runs **LEFT**, he stores "potential energy" in a credit counter.
3.  **Quadratic Speed Multiplier**: This credit then **multiplies** his future horizontal rewards. 
    - This creates a market proof for the agent: "Backtracking results in 10x more points per second once I turn back around."
4.  **Quadratic Velocity**: We reward `speed squared`. This makes the difference between "regular speed" and "Sonic speed" much more dramatic in the logs.

**Lesson for Students**: If your agent fails a physics-based challenge, don't just reward the result (passing). Reward the **preliminary actions** (Backtracking, Revving) that make the result possible!
