import sys
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

import torch
import argparse
import cv2

from src.utils import make_env
from src.agent import Agent

def evaluate(model_path=None):
    """
    Runs the agent in 'Inference Mode' (Play Mode).
    Unlike training, this doesn't update any weights. It just runs the game 
    so you can watch the agent play.
    """
    gym_id = "SonicTheHedgehog2-Genesis"
    state = "EmeraldHillZone.Act1"
    
    # Setup Environment
    env = make_env(gym_id, state)()
    
    # We need a dummy object to mimic the vector env structure for the agent
    class DummyEnvs:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
            
    dummy_envs = DummyEnvs(env)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(dummy_envs).to(device)
    
    if model_path:
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["agent_state_dict"])
        else:
            agent.load_state_dict(checkpoint)
    else:
        print("No model loaded, using random agent.")

    agent.eval()
    
    obs, _ = env.reset()
    # Add batch dimension (1, C, H, W)
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device, dtype=torch.float32)
    
    total_reward = 0
    steps = 0
    
    print("--- WATCHING THE AGENT (Press 'q' in the window to stop) ---")
    
    try:
        # [IMPORTANT] Inference Mode
        # This tells PyTorch: "We are not training, so don't calculate gradients."
        # This makes the code run much faster and uses less memory.
        with torch.inference_mode():
            while steps < 8000: # Limit length
                # Thinking
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                cpu_action = action.cpu().numpy()[0]
                
                # Action
                next_obs, reward, terminated, truncated, info = env.step(cpu_action)
                
                # Rendering - Always update window
                frame = env.unwrapped.get_screen()
                cv2.imshow("Sonic 2", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                # We must call waitKey(1) every frame if we want accurate timing on Windows
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                total_reward += reward
                steps += 1
                
                # Faster tensor conversion
                obs_tensor = torch.as_tensor(next_obs, device=device).unsqueeze(0).float()
                
                if steps % 50 == 0:
                    vx = info.get('x', 0) - info.get('prev_x', info.get('x', 0))
                    print(f"Step {steps} | Pos: ({info.get('x', 0)}, {info.get('y', 0)}) | VX: {vx} | Act: {cpu_action} | Rings: {info.get('rings', 0)}")

                if info.get('level_end_bonus', 0) > 0:
                    print(f"--- SIGNPOST HIT! (X={info.get('x', 0)}) ---")
                    print(f"Episode Victory! Reward: {total_reward:.2f}")
                    print("Freezing for 3 seconds...")
                    cv2.waitKey(3000)
                    obs, _ = env.reset()
                    obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0).float()
                    total_reward = 0
                    steps = 0
                    continue

                if terminated or truncated:
                    reason = "DIE" if terminated else "TIMEOUT/STUCK"
                    print(f"Episode Ended | Reason: {reason} | Final Pos: ({info.get('x', 0)}, {info.get('y', 0)}) | Total Reward: {total_reward:.2f}")
                    obs, _ = env.reset()
                    obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0).float()
                    total_reward = 0
                    steps = 0
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    
    evaluate(args.model)
