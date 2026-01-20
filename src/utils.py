import gymnasium as gym
import sys
import time
import random
import retro
import numpy as np

# We import our custom wrappers to apply them to the base retro environment
from src.env_wrappers import (
    SonicDiscretizer, 
    ResizeObservation, 
    PyTorchFrameStack, 
    RetroCompatibility, 
    TransposeObservation, 
    InfoRenderWrapper, 
    SonicRewardV0,
    TimeLimitWrapper,
    StagnationWrapper
)

def make_env(game, state, stack_frames=4, render=False):
    """
    Environment Factory.
    Creates a single instance of the Sonic environment with all necessary wrappers.
    This is called by the parallel process worker to initialize each game instance.
    """
    def _init():
        # --- WORKER-LEVEL COMPATIBILITY ---
        # Each worker process needs to have the 'gym' alias set correctly for Retro.
        import gymnasium as gym
        import sys
        sys.modules["gym"] = gym
        
        # Monkeypatch the seeding utility to work with newer gymnasium versions
        import gymnasium.utils.seeding as seeding
        def hash_seed(seed=None, max_bytes=8):
            if seed is None:
                seed = np.random.randint(0, 2**31 - 1)
            return int(seed)
        seeding.hash_seed = hash_seed
        sys.modules["gym.utils.seeding"] = seeding

        # --- WINDOWS STABILITY ---
        # Stagger start to avoid retro initialization race conditions.
        # Without this, multiple Genesis emulators trying to start at once can crash on Windows.
        time.sleep(random.random() * 2) 
        
        # Initialize base Retro environment
        env = retro.make(game=game, state=state)
        
        # Apply Wrappers in order:
        # 1. Compatibility: Sync return values with Gymnasium standards
        env = RetroCompatibility(env)
        
        # 2. Reward Shaping: Define what the agent should care about (Speed & Survival)
        env = SonicRewardV0(env)
        
        # 3. Visualization: (Optional) Pass frames back to main process for rendering
        if render:
            env = InfoRenderWrapper(env)
            
        # 4. Discretizer: Convert complex 12-button combo to 7 logical game commands
        env = SonicDiscretizer(env)
        
        # 5. Image Processing: Resize 2D pixels and transpose to PyTorch Tensor format (C, H, W)
        env = ResizeObservation(env, 84)
        env = TransposeObservation(env)
        
        # 6. Time Limit: Force restart after 3 minutes (10,800 frames)
        env = TimeLimitWrapper(env, max_steps=10800)
        
        # 7. Stagnation Check: Restart if Sonic is stuck for 30 seconds (1800 frames)
        env = StagnationWrapper(env, max_stagnant_steps=1800)
        
        # 8. Frame Stacking: Let the agent see 'time' by stacking 4 consecutive frames
        env = PyTorchFrameStack(env, stack_frames)
        
        return env
        
    return _init
