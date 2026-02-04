import gymnasium as gym
import sys
import torch
import numpy as np

# --- CRITICAL RETRO COMPATIBILITY ---
sys.modules["gym"] = gym
import gymnasium.utils.seeding as seeding
def hash_seed(seed=None, max_bytes=8):
    if seed is None: seed = np.random.randint(0, 2**31 - 1)
    return int(seed)
seeding.hash_seed = hash_seed
sys.modules["gym.utils.seeding"] = seeding

from src.utils import make_env
from src.agent import Agent

def debug_run(model_path):
    gym_id = "SonicTheHedgehog2-Genesis"
    state = "EmeraldHillZone.Act1"
    
    env_fn = make_env(gym_id, state)
    env = env_fn()
    
    class DummyEnvs:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(DummyEnvs(env)).to(device)
    
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    else:
        agent.load_state_dict(checkpoint)
    agent.eval()
    
    obs, info = env.reset()
    done = False
    step = 0
    
    print("Starting debug run. Logging frames when X > 8000...")
    
    last_x = 0
    while not done:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(device))
        
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
        done = terminated or truncated
        
        curr_x = info.get('x', 0)
        curr_y = info.get('y', 0)
        rings = info.get('rings', 0)
        lives = info.get('lives', 0)
        
        if curr_x > 8000:
            print(f"Frame {step}: X={curr_x}, Y={curr_y}, Rings={rings}, Lives={lives}, Reward={reward:.4f}")
            if curr_x < last_x - 100:
                print("!!! DEATH OR TELEPORT DETECTED !!!")
                # Log 10 more frames then stop
                for _ in range(10):
                    print(f"Post-Death Frame {step}: X={curr_x}, Y={curr_y}")
                break
        
        last_x = curr_x
        step += 1
        if step > 10000: break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    debug_run(args.model)
