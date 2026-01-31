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

def visual_verify(model_path, targets=[2720, 4400, 6000]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visually verifying {model_path} targets: {targets}")
    
    env_fn = make_env("SonicTheHedgehog2-Genesis", "EmeraldHillZone.Act1")
    env = env_fn()
    
    agent = Agent(DummyEnvs(env)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"] if "agent_state_dict" in checkpoint else checkpoint)
    agent.eval()
    
    obs, info = env.reset()
    frames_saved = 0
    
    for step in range(5000):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
        curr_x = info.get('x', 0)
        
        # Check targets
        for t in targets:
            if curr_x >= t and curr_x < t + 50:
                frame = env.unwrapped.get_screen()
                filename = f"verify_x{t}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"Saved {filename} at X={curr_x}")
                targets.remove(t)
                frames_saved += 1
                break
        
        if terminated or truncated or not targets:
            break
            
    env.close()
    return frames_saved

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    visual_verify(args.model)
