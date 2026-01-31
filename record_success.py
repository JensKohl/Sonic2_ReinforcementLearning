import sys
import gymnasium
# Monkeypatch gym for stable-retro compatibility
sys.modules["gym"] = gymnasium

import torch
import numpy as np
import os
import retro
import cv2
from src.agent import Agent
from src.utils import make_env

class DummyEnvs:
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.is_vector_env = False

def record_successful_run(model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_path} to record success...")
    
    game = "SonicTheHedgehog2-Genesis"
    state = "EmeraldHillZone.Act1"
    
    # Create the env
    env_fn = make_env(game, state)
    env = env_fn()
    
    agent = Agent(DummyEnvs(env)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    else:
        agent.load_state_dict(checkpoint)
    agent.eval()
    
    # Video setup
    # Note: Using 'XVID' for AVI or 'mp4v' for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Default Sonic size is 320x224, but wrapped might be different. 
    # We use the raw screen from info for better quality.
    video_writer = None
    
    max_episodes = 5
    for ep in range(max_episodes):
        obs, info = env.reset()
        frames = []
        max_x = 0
        total_steps = 0
        
        print(f"Running Episode {ep+1}...")
        for _ in range(3000):
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                # Use sample to match training behavior
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            
            # Capture frame (from info if available, or resize obs)
            # make_env with default wrappers provides transposed (C, H, W) 84x84
            # We want the ORIGINAL screen for the user if possible.
            # RetroEnv usually has an internal screen.
            frame = env.unwrapped.get_screen() # Original 320x224
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            max_x = max(max_x, info.get('x', 0))
            total_steps += 1
            if terminated or truncated:
                break
        
        print(f"  Episode finished. Max X: {max_x}")
        if max_x > 5000:
            print(f"  SUCCESS! Saving video to {output_path}...")
            height, width, _ = frames[0].shape
            video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))
            for f in frames:
                video_writer.write(f)
            video_writer.release()
            print("  Video saved.")
            env.close()
            return True
            
    print("Could not find a successful run in 5 attempts.")
    env.close()
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the model checkpoint")
    parser.add_argument("--threshold", type=int, default=5000, help="X threshold for success")
    parser.add_argument("--output", default="success_run.mp4", help="Output video path")
    args = parser.parse_args()
    
    # We call a modified version of record_successful_run that takes threshold
    def record_successful_run_v2(model_path, output_path, threshold):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_path} to record success...")
        
        game = "SonicTheHedgehog2-Genesis"
        state = "EmeraldHillZone.Act1"
        
        env_fn = make_env(game, state)
        env = env_fn()
        
        agent = Agent(DummyEnvs(env)).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if "agent_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["agent_state_dict"])
        else:
            agent.load_state_dict(checkpoint)
        agent.eval()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        max_episodes = 20
        for ep in range(max_episodes):
            obs, info = env.reset()
            frames = []
            max_x = 0
            
            print(f"Running Episode {ep+1}...")
            for _ in range(3000):
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_tensor)
                
                obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
                frame = env.unwrapped.get_screen()
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                max_x = max(max_x, info.get('x', 0))
                if terminated or truncated:
                    break
            
            print(f"  Episode finished. Max X: {max_x}")
            if max_x > threshold:
                print(f"  SUCCESS! Saving video to {output_path}...")
                height, width, _ = frames[0].shape
                video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))
                for f in frames:
                    video_writer.write(f)
                video_writer.release()
                print("  Video saved.")
                env.close()
                return True
                
        print(f"Could not find a run > {threshold} in {max_episodes} attempts.")
        env.close()
        return False

    record_successful_run_v2(args.model, args.output, args.threshold)
