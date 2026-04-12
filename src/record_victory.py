import sys
import gymnasium
sys.modules["gym"] = gymnasium

import torch
import cv2
from src.agent import Agent
from src.utils import make_env

class DummyEnvs:
    """
    A simple wrapper to trick the Agent into thinking it's running in a 'Vector Environment'.
    The Agent expects to handle multiple games at once (Batch Size > 1), but here we only have 1 game.
    """
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space

def record_victory(model_path, output_path, threshold=10000, playback_fps=60.0):
    """
    Records a video of the agent playing the game until it wins (reaches the threshold X coordinate).
    It will try up to 100 times to get a good run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_path} to record victory (Threshold={threshold}, FPS={playback_fps})...")
    
    game = "SonicTheHedgehog2-Genesis"
    state = "EmeraldHillZone.Act1"
    
    # We use a custom env fn that DOES NOT have the TimeLimit if possible, 
    # or just a very high one for recording.
    env_fn = make_env(game, state)
    env = env_fn()
    
    agent = Agent(DummyEnvs(env)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    else:
        agent.load_state_dict(checkpoint)
        
    # [IMPORTANT] Switch to Evaluation Mode
    # This tells PyTorch to disable things like Dropout and BatchNorm, 
    # ensuring the AI behaves deterministically (same input = same output).
    agent.eval()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for ep in range(100): # 100 attempts to get a win
        obs, info = env.reset()
        frames = []
        max_x = 0
        victory_frames = 0
        is_victory = False
        
        print(f"Running Episode {ep+1}...")
        # High step limit for victory recording (4000 steps = 4+ minutes)
        for step in range(4000):
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            # Step the env
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            
            # Use raw screen for video
            frame = env.unwrapped.get_screen()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            curr_x = info.get('x', 0)
            max_x = max(max_x, curr_x)
            
            # If we hit the threshold, we keep going for 300 more frames (5 seconds at 60fps)
            if curr_x >= threshold and not is_victory:
                print(f"  Goal reached at X={curr_x}! Recording buffer...")
                is_victory = True
            
            if is_victory:
                victory_frames += 1
                if victory_frames > 300: # 5 second victory lap
                    break
            
            if (terminated or truncated) and not is_victory:
                # If it ended before the goal, this episode is a failure for recording
                break
        
        print(f"  Episode finished. Max X: {max_x}")
        if is_victory:
            print(f"  SUCCESS! Final video frames: {len(frames)}")
            height, width, _ = frames[0].shape
            video_writer = cv2.VideoWriter(output_path, fourcc, playback_fps, (width, height))
            for f in frames:
                video_writer.write(f)
            video_writer.release()
            print("  Video saved.")
            env.close()
            return True
            
    print(f"Could not find a run > {threshold} in attempts.")
    env.close()
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", default="final_victory_run_v2.mp4")
    parser.add_argument("--threshold", type=int, default=10000)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()
    record_victory(args.model, args.output, args.threshold, args.fps)
