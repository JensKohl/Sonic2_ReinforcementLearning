import gymnasium as gym
import sys
from src.utils import make_env

def test():
    """
    Smoke Test: Can we create the Sonic environment without crashing?
    This verifies that the ROM is found and the wrappers are working.
    """
    print("Testing make_env returns...")
    fn = make_env("SonicTheHedgehog2-Genesis", "EmeraldHillZone.Act1")
    env = fn()
    print(f"Env created: {type(env)}")
    obs, info = env.reset()
    print(f"Reset returned: obs={type(obs)}, info={type(info)}")
    
    action = env.action_space.sample()
    res = env.step(action)
    print(f"Step returned {len(res)} values")
    print(f"Values types: {[type(x) for x in res]}")
    env.close()

if __name__ == "__main__":
    test()
