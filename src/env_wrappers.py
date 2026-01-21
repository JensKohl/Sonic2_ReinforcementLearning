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

class SonicRewardV0(gym.Wrapper):
    """
    Custom Reward Shaping:
    Encourages speed-running while penalizing death.
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_lives = 3
        self.max_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_lives = info.get('lives', 3)
        self.max_x = info.get('x', 0)
        self.prev_x = self.max_x
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        curr_x = info.get('x', 0)
        progress_reward = max(0, curr_x - self.max_x)
        self.max_x = max(self.max_x, curr_x)
        
        # Time pressure
        time_penalty = -0.005
        
        # Survival
        curr_lives = info.get('lives', 3)
        life_penalty = -50.0 if curr_lives < self.prev_lives else 0.0
        self.prev_lives = curr_lives
        
        # Win condition
        win_bonus = 250.0 if curr_x > 10000 else 0.0
        if win_bonus > 0: terminated = True
            
        # Momentum Reward (SonicRewardV6 - Hill Mastery): 
        # 1. REMOVED Bias: Rewards speed in ANY direction equally.
        # This makes running back for momentum "free" for the agent.
        # 2. Jump Penalty: Discourages "jitter-jumping" at hills.
        velocity = curr_x - self.prev_x
        speed = abs(velocity)
        
        # Start with zero momentum reward
        momentum_reward = 0.0
        
        # Only reward meaningful movement (Anti-Farming Threshold)
        if speed > 2.0:
            # High speed multiplier (Turbo)
            multiplier = 2.0 if speed > 4.0 else 1.0
            # Unbiased reward (Speed * 0.06 is the goal rate)
            momentum_reward = speed * 0.06 * multiplier
                
        self.prev_x = curr_x

        # Jump Penalty: Deduct points for jumping (actions 2, 3, 8 in Discretizer)
        # This forces the agent to try running up hills instead of jumping at them.
        jump_penalty = -0.1 if action in [2, 3, 8] else 0.0
            
        custom_reward = progress_reward + momentum_reward + time_penalty + life_penalty + win_bonus + jump_penalty
        return obs, float(custom_reward * 0.01), terminated, truncated, info

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
        self.total_movement_in_window = 0
        self.max_x = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        curr_x = info.get('x', 0)
        
        # We track "Total Movement" (distance traveled)
        # This allows Sonic to run back and forth without stalling the timer.
        dist_moved = abs(curr_x - self.last_x)
        self.total_movement_in_window += dist_moved
        self.last_x = curr_x
        self.current_stagnant_steps += 1
            
        if self.current_stagnant_steps >= self.max_stagnant_steps:
            # If total distance moved in 30 seconds is less than 600 pixels (Turbo Update), he's stuck.
            # 600 pixels / 1800 steps = 0.33 pixels/step average (prevents wiggling)
            if self.total_movement_in_window < 600:
                truncated = True
            else:
                # Reset the window but keep going
                self.current_stagnant_steps = 0
                self.total_movement_in_window = 0
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_stagnant_steps = 0
        self.total_movement_in_window = 0
        obs, info = self.env.reset(**kwargs)
        self.last_x = info.get('x', 0)
        return obs, info
