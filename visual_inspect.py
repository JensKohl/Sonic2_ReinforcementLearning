import cv2
import torch
import torch.nn as nn
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import numpy as np
import retro
from src.agent import Agent
from src.utils import make_env

class DummyEnvs:
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.is_vector_env = False

def visual_inspect(model_path, target_x=6860):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inspecting obstacle at X={target_x} with {model_path}")
    
    env_fn = make_env("SonicTheHedgehog2-Genesis", "EmeraldHillZone.Act1")
    env = env_fn()
    
    agent = Agent(DummyEnvs(env)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"] if "agent_state_dict" in checkpoint else checkpoint)
    agent.eval()
    
    obs, info = env.reset()
    
    for step in range(5000):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
        curr_x = info.get('x', 0)
        
        if curr_x >= target_x:
            frame = env.unwrapped.get_screen()
            filename = f"inspect_x{target_x}.jpg"
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Saved {filename} at X={curr_x}")
            break
        
        if terminated or truncated:
            break
            
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--x", type=int, default=6860)
    args = parser.parse_args()
    visual_inspect(args.model, args.x)
