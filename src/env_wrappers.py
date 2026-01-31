import gymnasium as gym
import numpy as np
import cv2

class FrameSkip(gym.Wrapper):
    """
    Repeats the same action for 'skip' frames.
    This is standard in Atari/Retro RL because it helps the agent 
    build momentum and reduces the "jittery" behavior of picking 
    a new action every 1/60th of a second.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class RetroCompatibility(gym.Wrapper):
    """
    Acts like a translator between the old Retro emulator and the new Gym.
    It takes the 4 values returned by Retro and turns them into the 5 values 
    expected by modern Reinforcement Learning libraries.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done # Terminated means the game ended (Level won or Sonic died)
        truncated = False # Truncated means the time ran out (not used here)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # New Gym sends 'seed' and 'options', but old Retro doesn't know what those are.
        # We throw them away so the game doesn't crash.
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple): return obs
        return obs, {}

class SonicDiscretizer(gym.ActionWrapper):
    """
    Reduces the MultiBinary(12) Genesis controller to a few useful Discrete actions.
    This simplifies the search space for the RL agent significantly.
    """
    def __init__(self, env):
        super().__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [
            ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], 
            ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], 
            ['B'], ['DOWN', 'A'] # Added A as alternative Spin Dash button
        ]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action].copy()

class ResizeObservation(gym.ObservationWrapper):
    """
    Resizes the 320x224 Genesis output to a standard 84x84 square.
    This makes the input 7x smaller, which makes the AI 7x faster!
    """
    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        # We tell the AI that the screen size has changed to 84x84.
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # We use 'INTER_NEAREST' because it's the fastest way to resize images.
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_NEAREST)

class TransposeObservation(gym.ObservationWrapper):
    """Converts (H, W, C) to (C, H, W) to match PyTorch expectations."""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        offset = (2, 0, 1)
        return np.transpose(observation, offset)

class PyTorchFrameStack(gym.Wrapper):
    """
    Stacks k consecutive frames (usually 4) in one big pile.
    If the AI only sees 1 frame, it doesn't know if Sonic is moving.
    By seeing 4 frames at once, it can see Sonic's velocity and acceleration.
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = []
        shp = env.observation_space.shape
        # New shape will be (k * Channels, Height, Width)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        # When the game restarts, we fill the stack with the first frame 4 times.
        ob, info = self.env.reset(**kwargs)
        self.frames = [ob] * self.k
        return self._get_ob(), info

    def step(self, action):
        # Every step, we add the new frame and throw away the oldest one.
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        self.frames.pop(0)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        # Glue the 4 frames together into one big observation.
        return np.concatenate(self.frames, axis=0)

class InfoRenderWrapper(gym.Wrapper):
    """
    Periodic frame capture for visualization.
    Only captures every 'frequency' steps to minimize IPC bottleneck.
    """
    def __init__(self, env, frequency=16):
        super().__init__(env)
        self.frequency = frequency
        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        info["render_frame"] = self.env.unwrapped.get_screen() if self.step_count % self.frequency == 0 else None
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        self.step_count = 0
        if isinstance(res, tuple):
            obs, info = res
            info["render_frame"] = self.env.unwrapped.get_screen()
            return obs, info
        return res

class SonicRewardV13(gym.Wrapper):
    """
    SonicRewardV13 (The Balanced Progress Wrapper):
    Merges discovery, momentum, and altitude logic into a single stream.
    
    Fixes the "Jump Trap": Altitude reward is suppressed if the agent jumps 
    while at the base of a hill, preventing them from trading speed for height.
    """
    def __init__(self, env):
        super().__init__(env)
        self.visited_tiles = set()
        self.prev_lives = 3
        self.max_x = 0
        self.prev_x = 0
        self.backtrack_credit = 0.0
        self.min_y = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.visited_tiles = set()
        self.prev_lives = info.get('lives', 3)
        self.max_x = info.get('x', 0)
        self.prev_x = self.max_x
        self.backtrack_credit = 0.0
        self.min_y = info.get('y', None)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        curr_x = info.get('x', 0)
        curr_y = info.get('y', 0)
        velocity_x = curr_x - self.prev_x
        
        # 1. DISCOVERY (Exploration)
        discovery_bonus = 0.0
        tile = (curr_x // 16, curr_y // 16)
        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            discovery_bonus = 0.01 # Subtle hint to explore
            
        # 2. SPIN DASH (Preparation)
        spin_dash_reward = 0.0
        if action in [7, 9] and abs(velocity_x) < 1:
            spin_dash_reward = 0.05
            
        # 3. BACKTRACK & MOMENTUM
        in_ramp_zone = (2600 < curr_x < 2800) or (4000 < curr_x < 4400)
        if in_ramp_zone and velocity_x < -1:
            self.backtrack_credit = min(20.0, self.backtrack_credit + 0.5)
            
        speed = max(0, velocity_x)
        speed_factor = 1.0 + (self.backtrack_credit * 0.1)
        momentum_reward = (speed ** 2) * 0.02 * speed_factor
        
        # 4. HORIZONTAL PROGRESS (Main Objective)
        # Scaled up past the waterfall (x=2400) to ensure it's priority #1
        progress_mult = 2.0 if curr_x > 2400 else 1.0
        progress_reward = 0.0
        if curr_x > self.max_x:
            progress_reward = (curr_x - self.max_x) * (1.0 + self.backtrack_credit * 0.2) * progress_mult
            self.max_x = curr_x
            self.backtrack_credit = max(0, self.backtrack_credit - 1.0)
            
        # 5. ALTITUDE (Climbing)
        # FIX: Only reward height gain if NOT jumping (to force running up the ramp)
        # and only if moving forward.
        # Action 2, 3, 8 involve jumping (B/A buttons)
        altitude_reward = 0.0
        is_jumping = action in [2, 3, 8, 9] # Using discretizer indices
        if curr_x > 2000 and self.min_y is not None and not is_jumping:
             if curr_y < self.min_y and velocity_x > 0:
                 altitude_reward = (self.min_y - curr_y) * 0.5
                 self.min_y = curr_y
        elif self.min_y is None or (curr_x < self.prev_x - 5): # Reset min_y if we fell back
             self.min_y = curr_y
        
        # 6. PENALTIES
        curr_lives = info.get('lives', 3)
        life_penalty = -100.0 if curr_lives < self.prev_lives else 0.0
        self.prev_lives = curr_lives
        
        win_bonus = 500.0 if curr_x > 10000 else 0.0
        if win_bonus > 0: terminated = True

        self.prev_x = curr_x
        self.prev_y = curr_y

        total_custom = progress_reward + spin_dash_reward + momentum_reward + \
                       altitude_reward + discovery_bonus + life_penalty + win_bonus - 0.01
        
        return obs, float(total_custom * 0.01), terminated, truncated, info

class TimeLimitWrapper(gym.Wrapper):
    """
    Stops the episode after a fixed number of steps.
    This encourages the agent to find the finish line faster.
    """
    def __init__(self, env, max_steps=10800): # 10,800 steps = 3 minutes at 60 FPS
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

class StagnationWrapper(gym.Wrapper):
    """
    Truncates the episode if the agent's x-coordinate doesn't increase for a while.
    This prevents the agent from getting stuck behind obstacles for too long.
    """
    def __init__(self, env, max_stagnant_steps=1800): # 30 seconds at 60 FPS
        super().__init__(env)
        self.max_stagnant_steps = max_stagnant_steps
        self.current_stagnant_steps = 0
        self.last_x = 0
        self.max_x = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        curr_x = info.get('x', 0)
        
        # We track progress toward max_x
        self.last_x = curr_x
        self.current_stagnant_steps += 1
            
        if self.current_stagnant_steps >= self.max_stagnant_steps:
            # If max_x hasn't increased in 30 seconds, Sonic is stuck.
            # This is much stricter than movement tracking and prevents behavioral loops.
            if curr_x <= self.max_x:
                truncated = True
            else:
                # Progress was made, reset the window
                self.max_x = curr_x
                self.current_stagnant_steps = 0
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_stagnant_steps = 0
        obs, info = self.env.reset(**kwargs)
        self.last_x = info.get('x', 0)
        self.max_x = self.last_x
        return obs, info
