import os
import sys
import subprocess
import argparse
import gymnasium as gym

# --- CRITICAL RETRO COMPATIBILITY BLOCK ---
# We must alias gym to gymnasium before ANY other imports that use retro
sys.modules["gym"] = gym

import numpy as np
import gymnasium.utils.seeding as seeding
def hash_seed(seed=None, max_bytes=8):
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    return int(seed)
seeding.hash_seed = hash_seed
sys.modules["gym.utils.seeding"] = seeding
# ------------------------------------------

import time
import torch
import torch.optim as optim
import cv2
from torch.utils.tensorboard import SummaryWriter

from src.utils import make_env
from src.agent import Agent
from src.ppo import PPOAlgo, RolloutBuffer

class Callback:
    """
    Base class for training callbacks. 
    Callbacks allow us to "hook" into the training loop and perform actions 
    at specific moments (like every step or every network update).
    """
    def on_step(self, global_step, infos):
        """Called after every environment step."""
        pass
    def on_update(self, update, global_step, train_stats):
        """Called after a full PPO network update."""
        pass

class GPUTemperatureCallback(Callback):
    """
    Safety mechanism: Monitors the GPU temperature during training.
    If the GPU gets too hot (threshold), it stops training to protect the hardware.
    This is especially important on Windows laptops or single-GPU desktop setups.
    """
    def __init__(self, threshold=85, check_freq=10, writer=None):
        self.threshold = threshold
        self.check_freq = check_freq
        self.writer = writer
        self.last_check_update = 0

    def on_update(self, update, global_step, train_stats):
        if update % self.check_freq == 0:
            try:
                # Query GPU temperature using nvidia-smi
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                temp = int(result.stdout.strip())
                
                if self.writer:
                    self.writer.add_scalar("charts/gpu_temperature", temp, global_step)

                if temp >= self.threshold:
                    print(f"\n[CRITICAL] GPU temperature reached {temp}°C! Threshold is {self.threshold}°C.")
                    print("Stopping training to prevent overheating.")
                    raise KeyboardInterrupt("GPU too hot") # Using KeyboardInterrupt to stop clean-ish
                
            except subprocess.CalledProcessError:
                pass # Silently fail if nvidia-smi is not available or fails
            except Exception as e:
                print(f"Warning: GPU Temperature check failed: {e}")

class CheckpointCallback(Callback):
    """
    The "Save Game" system for AI training.
    Saves the entire state of the agent and optimizer periodically.
    This allows us to resume training if the computer crashes or if we want 
    to test an earlier version of the AI.
    """
    def __init__(self, save_freq, save_path, run_name):
        self.save_freq = save_freq
        self.save_path = save_path
        self.run_name = run_name
        os.makedirs(save_path, exist_ok=True)

    def save_checkpoint(self, update, global_step, agent, optimizer):
        state = {
            "update": update,
            "global_step": global_step,
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_name": self.run_name,
        }
        
        # Save numbered checkpoint
        path = os.path.join(self.save_path, f"{self.run_name}_step_{global_step}.pth")
        torch.save(state, path)
        
        # Also maintain a 'latest' symlink-like file (just a copy for robustness on Windows)
        latest_path = os.path.join(self.save_path, "latest_checkpoint.pth")
        torch.save(state, latest_path)
        
        # Also update a text file with the latest run name for easy discovery
        with open(os.path.join(self.save_path, "latest_run.txt"), "w") as f:
            f.write(self.run_name)

    def on_update(self, update, global_step, train_stats):
        if update % self.check_freq_updates() == 0: # Note: check_freq logic below
            pass
    
    def check_freq_updates(self):
        # We'll just hardcode 50 updates for now as in the original code's if update % 50 == 0
        return 50

    def on_update_with_data(self, update, global_step, agent, optimizer):
        if update % 50 == 0:
            self.save_checkpoint(update, global_step, agent, optimizer)

class BestModelCallback(Callback):
    """
    Hall of Fame system: Keeps track of the highest score (reward) achieved.
    Whenever the agent breaks its own record, this callback saves the current 
    model as the new "best.pth".
    """
    def __init__(self, save_path, run_name):
        self.save_path = save_path
        self.run_name = run_name
        self.best_reward = -float('inf')
        os.makedirs(save_path, exist_ok=True)

    def on_step(self, global_step, infos):
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    reward = info["episode"]["r"]
                    if reward > self.best_reward:
                        self.best_reward = reward
                        path = os.path.join(self.save_path, f"{self.run_name}_best.pth")
                        # Note: We need the agent state here. We'll pass it in train_stats or similar
                        pass 

    def update_best(self, agent, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            path = os.path.join(self.save_path, f"{self.run_name}_best.pth")
            torch.save(agent.state_dict(), path)
            return True
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Sonic 2 RL Training")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file to resume from")
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Hyperparameters
    exp_name = "Sonic2_PPO"
    gym_id = "SonicTheHedgehog2-Genesis"
    state = "EmeraldHillZone.Act1"
    
    total_timesteps = 15_000_000
    learning_rate = 2.5e-4
    num_envs = 8 # Optimal for RTX 2060
    num_steps = 512 # Balanced: 512 steps is large enough for efficiency but fast enough for frequent updates
    batch_size = num_envs * num_steps
    minibatch_size = 128 # Increased for better GPU utilization
    update_epochs = 4
    gamma = 0.99
    gae_lambda = 0.95
    
    # Global State Initialization
    global_step = 0
    start_update = 1
    run_name = f"{exp_name}__{int(time.time())}"
    checkpoint_path = None

    # Handle Resuming
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.resume:
        latest_cp = "models/checkpoints/latest_checkpoint.pth"
        if os.path.exists(latest_cp):
            checkpoint_path = latest_cp
        else:
            print("Warning: --resume flag set but no latest_checkpoint.pth found. Starting from scratch.")

    checkpoint_data = None
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        loaded_data = torch.load(checkpoint_path)
        
        if isinstance(loaded_data, dict) and "global_step" in loaded_data:
            # Full Checkpoint
            checkpoint_data = loaded_data
            global_step = checkpoint_data["global_step"]
            start_update = checkpoint_data["update"] + 1
            run_name = checkpoint_data["run_name"]
            print(f"Resuming run '{run_name}' at update {start_update}, global step {global_step}")
        else:
            # Weight-only checkpoint (e.g. best.pth)
            print("Detected weight-only checkpoint. Starting fine-tuning run.")
            # We treat the loaded data as the state dict
            checkpoint_data = loaded_data
            # Start fresh counters
            run_name = f"{exp_name}_finetune_{int(time.time())}"

    # Setup
    writer = SummaryWriter(f"logs/{run_name}", purge_step=global_step if (checkpoint_data and "global_step" in checkpoint_data) else None)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Vectorized Environments
    envs = gym.vector.AsyncVectorEnv(
        [make_env(gym_id, state, render=(i == 0)) for i in range(num_envs)]
    )
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    # Load state dicts if resuming
    if checkpoint_data:
        if isinstance(checkpoint_data, dict) and "agent_state_dict" in checkpoint_data:
            agent.load_state_dict(checkpoint_data["agent_state_dict"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        else:
            print("Loading weights from state dictionary...")
            agent.load_state_dict(checkpoint_data)

    algo = PPOAlgo(agent, optimizer, device, ent_coef=0.01)
    buffer = RolloutBuffer(num_steps, num_envs, envs.single_observation_space.shape, envs.single_action_space.shape, device)

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="models/checkpoints", run_name=run_name)
    best_model_callback = BestModelCallback(save_path="models", run_name=run_name)
    # GPU Temperature Safety Callback (Threshold: 85°C, check every 20 updates)
    temp_callback = GPUTemperatureCallback(threshold=85, check_freq=20, writer=writer)

    start_time = time.time()
    
    try:
        obs, _ = envs.reset()
        obs = torch.as_tensor(obs, device=device).float()
        
        num_updates = total_timesteps // batch_size
        print(f"Starting training for {num_updates} updates (Current: {start_update}/{num_updates})...")
        
        from tqdm import tqdm
        for update in tqdm(range(start_update, num_updates + 1), desc="Training"):
            iteration_start_time = time.time()
            buffer.reset()
            
            # Anneal learning rate: We slowly decrease the step size over time.
            # This helps the agent "settle down" into an optimal policy as it learns.
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

            # --- ROLLOUT PHASE ---
            # The agent plays the game for 'num_steps' and stores the results.
            for step in range(num_steps):
                global_step += 1 * num_envs # Track total frames processed across all environments
                
                # Get action from the policy (no gradients needed during data collection)
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs)
                
                # Convert tensor action to numpy to send to the Genesis emulator
                cpu_action = action.cpu().numpy()
                
                # Step the environment: This is where the physics and game logic happen.
                next_obs, rewards, terminated, truncated, infos = envs.step(cpu_action)
                done = np.logical_or(terminated, truncated) # Combine termination (death/win) and truncation (time)
                
                # --- RENDERING (Visualization) ---
                # We only render Env 0 to save performance.
                if "render_frame" in infos:
                    frame = infos["render_frame"][0]
                    if frame is not None:
                        # Convert RGB to BGR for OpenCV display
                        cv2.imshow("Sonic 2 Training (Env 0)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        
                        # [SPEED HACK]: We only call waitKey(1) when we actually update the window.
                        # Calling waitKey(1) every step adds roughly 1-2 seconds of total delay per iteration!
                        cv2.waitKey(1)
                
                # Optimization: Convert to tensor and move to GPU (non-blocking)
                # Note: this requires obs to be a numpy array from AsyncVectorEnv
                buffer.add(obs, action, logprob, rewards, done, value)
                obs = torch.from_numpy(next_obs).to(device, non_blocking=True).float()
                
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            ep_reward = float(info['episode']['r'])
                            # Log to Tensorboard: Great for long-term graphing
                            writer.add_scalar("charts/episodic_return", ep_reward, global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            
                            # Log actual victory (signpost reached)
                            is_victory = 1.0 if info.get("level_end_bonus", 0) > 0 else 0.0
                            writer.add_scalar("charts/victory", is_victory, global_step)
                            
                            # Save the best model automatically based on high-score
                            if best_model_callback.update_best(agent, ep_reward):
                                tqdm.write(f"--> [NEW BEST] Step: {global_step} | Reward: {ep_reward:.2f}")

            # --- OPTIMIZATION PHASE ---
            # After collecting a rollout, we update our neural network.
            
            # Bootstrap value: Estimate what the future rewards look like from the current state
            with torch.no_grad():
                next_value = agent.get_value(obs).reshape(-1)
                
            # Compute Returns and Advantages (the target targets for our network)
            buffer.returns, buffer.advantages = buffer.compute_returns_and_advantages(
                next_value, torch.from_numpy(done).to(device).float(), gamma, gae_lambda
            )
            
            # Policy Update: Adjust weights using the Adam optimizer
            train_stats = algo.update(buffer, minibatch_size, update_epochs)
            train_stats['agent_state'] = agent.state_dict() # Capture state for any potential checkpoint
            
            # SPS & Metrics Update
            current_time = time.time()
            sps = int(batch_size / (current_time - iteration_start_time))
            iteration_start_time = current_time
            
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", train_stats["v_loss"], global_step)
            writer.add_scalar("losses/policy_loss", train_stats["pg_loss"], global_step)
            writer.add_scalar("losses/entropy", train_stats["entropy"], global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            
            # Periodic Console Summary
            if update % 20 == 0 or update == 1:
                tqdm.write(
                    f"\n--- Update {update}/{num_updates} ---\n"
                    f"Global Step: {global_step}\n"
                    f"SPS: {sps}\n"
                    f"Policy Loss: {train_stats['pg_loss']:.4f}\n"
                    f"Value Loss: {train_stats['v_loss']:.4f}\n"
                    f"Entropy: {train_stats['entropy']:.4f}\n"
                    f"-------------------------"
                )

            # Periodic Checkpoint
            checkpoint_callback.on_update_with_data(update, global_step, agent, optimizer)
            
            # GPU Temperature Check
            temp_callback.on_update(update, global_step, train_stats)
            
    finally:
        # Save Final Model
        os.makedirs("models", exist_ok=True)
        torch.save(agent.state_dict(), f"models/{run_name}_final.pth")
        print(f"Final model saved as {run_name}_final.pth")
        envs.close()
        writer.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    train()
