import gymnasium as gym
import sys
import torch
import numpy as np

# Retro compatibility
sys.modules["gym"] = gym
import gymnasium.utils.seeding as seeding
def hash_seed(seed=None, max_bytes=8):
    if seed is None: seed = np.random.randint(0, 2**31 - 1)
    return int(seed)
seeding.hash_seed = hash_seed
sys.modules["gym.utils.seeding"] = seeding

from src.utils import make_env
from src.agent import Agent

def check_progress(model_path):
    """
    Runs a "Reliability Test" for the AI.
    Instead of just running once, we run 10 episodes and count how many times 
    the agent successfully passes a certain point (e.g., 5000 pixels).
    This gives us a "Success Rate" (e.g., 80% reliable).
    """
    gym_id = "SonicTheHedgehog2-Genesis"
    state = "EmeraldHillZone.Act1"
    
    # Create the environment with all our custom wrappers
    env_fn = make_env(gym_id, state)
    env = env_fn()
    
    # Dummy wrapper to make the single env look like a vector of envs
    class DummyEnvs:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = Agent(DummyEnvs(env)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    else:
        agent.load_state_dict(checkpoint)
    agent.eval()
    
    num_episodes = 10
    print(f"Testing reliability of {model_path} over {num_episodes} episodes...")
    successes = 0
    max_xs = []
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0).float()
        episode_max_x = 0
        
        for step in range(2500):
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            curr_x = info.get('x', 0)
            episode_max_x = max(episode_max_x, curr_x)
            obs_tensor = torch.as_tensor(next_obs, device=device).unsqueeze(0).float()
            
            if terminated or truncated:
                break
        
        # Update success threshold to something meaningful (e.g. 5000 = midway, 10000 = signpost)
        passed = episode_max_x > 5000
        if passed: successes += 1
        max_xs.append(episode_max_x)
        print(f"  Episode {episode}: Max X = {episode_max_x} {'[PASSED]' if passed else '[FAILED]'}")
        
    # Calculate Final Stats
    # AVG_MAX_X tells us "on average, how far does it get?"
    # SUCCESS_RATE tells us "how often does it beat the level?"

    print(f"DIAGNOSTIC_RESULT: SUCCESS_RATE={successes/num_episodes*100}% | AVG_MAX_X={sum(max_xs)/num_episodes}")
    env.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_progress(sys.argv[1])
    else:
        print("Please provide model path")
