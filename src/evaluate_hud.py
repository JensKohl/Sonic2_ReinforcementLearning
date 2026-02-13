import os
import sys
import gymnasium as gym
import numpy as np
import cv2
import torch
import time
import argparse

# --- CRITICAL RETRO COMPATIBILITY BLOCK ---
# Ensure gym is aliased to gymnasium for retro compatibility
sys.modules["gym"] = gym
import gymnasium.utils.seeding as seeding
def hash_seed(seed=None, max_bytes=8):
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    return int(seed)
seeding.hash_seed = hash_seed
sys.modules["gym.utils.seeding"] = seeding
# ------------------------------------------

from src.agent import Agent

# We define a local version of make_env to allow custom limits without modifying src/utils.py
from src.env_wrappers import (
    SonicDiscretizer, ResizeObservation, PyTorchFrameStack, RetroCompatibility,
    TransposeObservation, InfoRenderWrapper, TimeLimitWrapper,
    StagnationWrapper, FrameSkip, SonicRewardV17
)
import retro

def make_env_eval(game, state, stack_frames=4, render=False, max_steps=None, max_stagnant_steps=None):
    def _init():
         # --- WORKER-LEVEL COMPATIBILITY ---
        import gymnasium as gym
        import sys
        sys.modules["gym"] = gym
        import gymnasium.utils.seeding as seeding
        def hash_seed(seed=None, max_bytes=8):
            if seed is None:
                seed = np.random.randint(0, 2**31 - 1)
            return int(seed)
        seeding.hash_seed = hash_seed
        sys.modules["gym.utils.seeding"] = seeding
        
        time.sleep(0.1) # Small delay
        
        env = retro.make(game=game, state=state)
        env = RetroCompatibility(env)
        env = FrameSkip(env, skip=3)
        env = SonicDiscretizer(env)
        env = SonicRewardV17(env)
        
        if render:
            env = InfoRenderWrapper(env)
            
        env = ResizeObservation(env, 84)
        env = TransposeObservation(env)
        
        # Use a local variable to avoid shadowing the outer 'max_steps' argument
        limit_steps = max_steps
        if limit_steps is None:
            limit_steps = 5400 # 6 minutes
        env = TimeLimitWrapper(env, max_steps=limit_steps)
        if max_stagnant_steps:
            env = StagnationWrapper(env, max_stagnant_steps=max_stagnant_steps)
            
        env = PyTorchFrameStack(env, stack_frames)

        # 9. Statistics (Capture EVERYTHING below, including Timeouts/Stagnation)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        return env
    return _init


# Action Labels matching SonicDiscretizer in env_wrappers.py
ACTION_LABELS = [
    "LEFT",      # 0
    "RIGHT",     # 1
    "LEFT+J",    # 2 (Jump Left)
    "RIGHT+J",   # 3 (Jump Right)
    "LEFT+D",    # 4 (Roll Left)
    "RIGHT+D",   # 5 (Roll Right)
    "DOWN",      # 6
    "DOWN+J",    # 7 (Spin Dash Charge)
    "JUMP",      # 8
    "SPIN(A)"    # 9 (Alt Spin Dash)
]

def draw_info_panel(frame, action_index, reward_sum, value_est=None, episode_num=1):
    """
    Draws a larger information panel at the bottom of the frame.
    """
    h, w, c = frame.shape
    
    # Define panel dimensions
    panel_height = 80
    panel_color = (30, 30, 30) # Dark gray background
    
    # Create the canvas with extra space at the bottom
    canvas = cv2.copyMakeBorder(frame, 0, panel_height, 0, 0, cv2.BORDER_CONSTANT, value=panel_color)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    
    # Colors
    col_active = (0, 255, 0)     # Bright Green
    col_inactive = (100, 100, 100) # Dim Gray
    col_text = (220, 220, 220)   # Off-white
    
    # Layout Config
    start_x = 10
    start_y = h + 20
    col_spacing = 65
    row_spacing = 25
    
    # --- DRAW ACTIONS ---
    # Row 1: Actions 0-4
    for i in range(5):
        label = ACTION_LABELS[i]
        color = col_active if i == action_index else col_inactive
        # Make active action bolder
        thick = 2 if i == action_index else 1
        cv2.putText(canvas, label, (start_x + i * col_spacing, start_y), font, font_scale, color, thick)
        
    # Row 2: Actions 5-9
    for i in range(5, 10):
        label = ACTION_LABELS[i]
        color = col_active if i == action_index else col_inactive
        thick = 2 if i == action_index else 1
        cv2.putText(canvas, label, (start_x + (i - 5) * col_spacing, start_y + row_spacing), font, font_scale, color, thick)
        
    # --- DRAW STATS ---
    stats_x = 10
    stats_y = start_y + row_spacing * 2
    
    # Stats Line
    stats_text = f"Ep: {episode_num} | Rew: {reward_sum:.1f}"
    cv2.putText(canvas, stats_text, (stats_x, stats_y), font, font_scale, col_text, 1)
    
    # Critic Value (if provided)
    if value_est is not None:
        val_text = f"Val: {value_est:.2f}"
        cv2.putText(canvas, val_text, (stats_x + 180, stats_y), font, font_scale, (0, 200, 255), 1)

    return canvas

def evaluate_with_hud(args):
    """
    Main loop for running the agent with visualization.
    """
    model_path = args.model
    stochastic_mode = args.stochastic

    gym_id = "SonicTheHedgehog2-Genesis"
    state = "EmeraldHillZone.Act1"
    
    print(f"--- SONIC 2 HUD EVALUATION ---")
    print(f"Model: {model_path}")
    print(f"Press 'Q' or 'ESC' to quit.")
    
    # Create Environment
    # render=True allows us to get the raw frame if needed, but we use get_screen() usually
    # Relax stagnation check for evaluation (boss fights can take time stationary)
    max_steps = None if args.infinite else 5400 # 6 minutes
    max_stagnant = None if args.infinite else 3600 # 3 minutes (Was 450 / 22s)
    
    env = make_env_eval(gym_id, state, render=False, max_steps=max_steps, max_stagnant_steps=max_stagnant)() 
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize Agent
    # We need a dummy env object to pass to Agent init to get shapes right
    class DummyEnv:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    agent = Agent(DummyEnv(env)).to(device)
    
    # Load Model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle both full checkpoint and state_dict only
        if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["agent_state_dict"])
        else:
            agent.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    agent.eval()
    
    # Evaluation Loop
    obs, _ = env.reset()
    # Convert obs to tensor: (1, 4, 84, 84)
    obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float()
    
    episode_count = 1
    total_reward = 0.0
    step_counter = 0
    victory_cooldown = 0
    
    try:
        with torch.no_grad():
            while True:
                # 1. Agent Decision
                if args.stochastic:
                    # Sample from distribution (randomness included)
                    action, _, _, value = agent.get_action_and_value(obs_tensor)
                    action_scalar = action.item()
                    value_scalar = value.item()
                else:
                    # Deterministic: Argmax of logits (Best possible move)
                    hidden = agent.network(obs_tensor)
                    logits = agent.actor(hidden)
                    action_scalar = torch.argmax(logits, dim=1).item()
                    value = agent.critic(hidden)
                    value_scalar = value.item()
                
                # 2. Step Environment
                next_obs, reward, terminated, truncated, info = env.step(action_scalar)
                total_reward += reward

                # Get curr_x from info immediately so we can check victory condition
                curr_x = info.get('x', 0)
                curr_y = info.get('y', 0)

                # Log position periodically to see where it gets stuck
                step_counter += 1
                if step_counter % 60 == 0:
                    bonus = info.get('level_end_bonus', -1)
                    print(f"Step {step_counter} | X: {curr_x} | Y: {curr_y} | Rings: {info.get('rings', -1)} | Bonus: {bonus}")

                # 3. Visualization
                # Get the raw screen from the emulator for the human to see (320x224)
                # The 'env' is wrapped, so we need to dig for the original retro env or use render()
                # simplest is typically env.unwrapped.get_screen() if available
                raw_screen = env.unwrapped.get_screen()
                
                # Convert to BGR for OpenCV
                frame_bgr = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2BGR)
                
                # Draw HUD
                hud_frame = draw_info_panel(frame_bgr, action_scalar, total_reward, value_scalar, episode_count)
                
                # Scale up for easier viewing (2x)
                display_frame = cv2.resize(hud_frame, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow("Sonic 2 Agent View", display_frame)
                
                # Input Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # Q or ESC
                    print("Quitting...")
                    break
                
                # 4. Prepare next step
                # Standardized Victory Detection (Signpost Hit):
                # Beginners: In Sonic 2, reaching the goal is signaled by a 'bonus' start.
                if info.get('level_end_bonus', 0) > 0:
                    print(f"--- SIGNPOST HIT! (X={curr_x}) ---")
                    # Overlay "LEVEL CLEAR" immediately
                    h, w, _ = display_frame.shape
                    cv2.putText(display_frame, "LEVEL CLEAR!", (w//2 - 140, h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow("Sonic 2 Agent View", display_frame)
                    
                    print(f"Episode {episode_count} Victory! Reward: {total_reward:.2f}")
                    print("Freezing for 3 seconds...")
                    cv2.waitKey(3000)
                    
                    # Clean Reset for next episode
                    obs, _ = env.reset()
                    obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float()
                    total_reward = 0.0
                    episode_count += 1
                    step_counter = 0
                    continue

                # Handle other terminations (Death / Time)
                if terminated or truncated:
                    print(f"Episode {episode_count} End (Terminated/Truncated). Reward: {total_reward:.2f}")
                    obs, _ = env.reset()
                    obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float()
                    total_reward = 0.0
                    episode_count += 1
                    step_counter = 0
                else:
                    obs_tensor = torch.tensor(next_obs, device=device).unsqueeze(0).float()
                    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sonic 2 PPO Agent with HUD")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pth model file")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic sampling (randomness) instead of deterministic argmax")
    parser.add_argument("--infinite", action="store_true", help="Disable time limits and stagnation checks")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)
        
    evaluate_with_hud(args)
