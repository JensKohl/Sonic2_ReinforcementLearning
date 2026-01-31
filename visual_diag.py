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

def run_visual_diag(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_path} for visual diagnosis on {device}...")
    
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
    
    num_episodes = 3
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        max_x = 0
        reached_waterfall = False
        
        print(f"\n--- Episode {ep} ---")
        for step in range(3000):
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            # Use sample() to match training stochasticity
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            
            curr_x = info.get('x', 0)
            curr_y = info.get('y', 0)
            if curr_x > max_x: max_x = curr_x
            
            # Log specific milestones
            if not reached_waterfall and curr_x > 2400:
                print(f"  [Step {step}] Reached Waterfall Zone! (X={curr_x}, Y={curr_y})")
                reached_waterfall = True
            
            # Log action probabilities when at the ramp area
            if 2600 < curr_x < 2800 and step % 50 == 0:
                print(f"  [Step {step}] At Ramp Area: X={curr_x}, Y={curr_y}")
                with torch.no_grad():
                    logits = agent.actor(obs_tensor)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    top_actions = np.argsort(probs)[-3:][::-1]
                    action_str = ", ".join([f"Act {a}: {probs[a]:.2f}" for a in top_actions])
                    print(f"    Probabilities: {action_str}")

            if terminated or truncated:
                break
        
        print(f"Episode {ep} Finished. Max X: {max_x} | Final Y: {info.get('y', 0)}")
    
    env.close()

if __name__ == "__main__":
    run_visual_diag("models/checkpoints/latest_checkpoint.pth")
