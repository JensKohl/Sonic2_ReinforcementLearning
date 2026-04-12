"""
Environment Wrappers for shaping the Sonic environment.
"""
import gymnasium as gym
import numpy as np
import cv2

# =============================================================================
# PREPROCESSING WRAPPERS
# =============================================================================


class FrameSkip(gym.Wrapper):
    """
    #### Wrapper: FrameSkip
    **Concept**: Temporal Abstraction.

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
            # We perform the same action multiple times.
            # In Sonic, physics (velocity) takes several frames to build up,
            # so acting every single frame (at 60Hz) is too fast for the AI
            # to "see" the results of its actions clearly.
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            # If the episode ends mid-skip, we stop immediately.
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class RetroCompatibility(gym.Wrapper):
    """
    Acts like a translator between the old Retro emulator and the new Gym.
    It takes the 4 values returned by Retro and turns them into the 5 values
    expected by modern Reinforcement Learning libraries.
    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # 'terminated' = Sonic died or beat the level.
        # 'truncated' = Time limit hit (handled by a separate wrapper).
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # New Gym sends 'seed' and 'options', but old Retro doesn't know what those are.
        # We throw them away so the game doesn't crash.
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        obs = self.env.reset(**kwargs)
        # Ensure we always return (observation, info_dict)
        if isinstance(obs, tuple):
            return obs
        return obs, {}


class SonicDiscretizer(gym.ActionWrapper):
    """
    #### Wrapper: SonicDiscretizer
    **Concept**: Action Space Reduction.

    The Genesis has 12 buttons. The AI could theoretically press any combination (2^12 = 4096).
    Most of these (like UP+DOWN or START+A) are useless or invalid.
    Hence, this simplifies the search space for the RL agent significantly.

    We map the 4000+ possibilities down to ~10 logical "Game Intents":
    - Move Left / Move Right
    - Jump Left / Jump Right
    - Spin Dash / Crouch / Jump in place

    This makes it 400x easier for the AI to find a good policy.
    """

    def __init__(self, env):
        super().__init__(env)
        buttons = [
            "B",
            "A",
            "MODE",
            "START",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "C",
            "Y",
            "X",
            "Z",
        ]
        actions = [
            ["LEFT"],
            ["RIGHT"],  # Basic movement
            ["LEFT", "B"],
            ["RIGHT", "B"],  # Jumping with direction
            ["LEFT", "DOWN"],
            ["RIGHT", "DOWN"],  # Ducking while moving (rolling)
            ["DOWN"],  # Crouch
            ["DOWN", "B"],  # Spin Dash Charge
            ["B"],  # Jump in place
            ["DOWN", "A"],  # Alt Spin Dash button
        ]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        # Tell Gymnasium we now only have 10-11 possible discrete actions.
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action].copy()


class ResizeObservation(gym.ObservationWrapper):
    """
    #### Wrapper: ResizeObservation
    **Concept**: Dimensionality Reduction.

    The raw Genesis image is 320x224 pixels in full color.
    That's 215,040 numbers the AI has to process every frame.
    By resizing to 84x84, we reduce the data by ~90% without losing
    essential information (like obstacles or platforms).
    """

    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        # Update the observation space metadata so the AI knows to expect 84x84.
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        # CV2 handles the pixel resampling.
        # 'INTER_NEAREST' is fastest way to resize images while preserving pixel sharpess.
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_NEAREST)


class TransposeObservation(gym.ObservationWrapper):
    """
    #### Wrapper: TransposeObservation
    **Concept**: Tensor Formatting.

    OpenCV/Gym use (Height, Width, Channels).
    PyTorch expects (Channels, Height, Width).
    This wrapper swaps the axes so the Deep Learning model can read the image.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, observation):
        offset = (2, 0, 1)
        return np.transpose(observation, offset)


class PyTorchFrameStack(gym.Wrapper):
    """
    #### Wrapper: PyTorchFrameStack
    **Concept**: Velocity Encoding.

    A single static image doesn't tell you if Sonic is moving.
    By stacking the last 4 frames together, the AI can "see" motion across time.
    Imagine a flip-book: seeing 4 pages at once lets you perceive the trajectory of a jump.
    """

    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = []
        shp = env.observation_space.shape
        # The new input channel count is (Original Channels * k)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0] * k, shp[1], shp[2]),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        # On reset, we don't have history yet, so we stack the first frame k times.
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_ob(), info

    def step(self, action):
        # Update the stack: pop the oldest, push the newest.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        self.frames.pop(0)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.concatenate(self.frames, axis=0)


class InfoRenderWrapper(gym.Wrapper):
    """
    #### Wrapper: InfoRenderWrapper
    **Concept**: Debug Visualization.

    Periodically saves the high-resolution screen into the 'info' dict.
    This allows us to record high-quality video without slowing down the AI,
    because we only "take a picture" every 16 steps.
    """

    def __init__(self, env, frequency=16):
        super().__init__(env)
        self.frequency = frequency
        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        # Extract the raw screen from the emulator core.
        info["render_frame"] = (
            self.env.unwrapped.get_screen()
            if self.step_count % self.frequency == 0
            else None
        )
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        self.step_count = 0
        if isinstance(res, tuple):
            obs, info = res
            info["render_frame"] = self.env.unwrapped.get_screen()
            return obs, info
        return res


class SonicRewardV18(gym.Wrapper):
    """
    #### Wrapper: SonicRewardV18 (Anti-Farming & Physics Logic)
    **Concept**: Reward Engineering for Physics-Based Games.

    This is the definitive version of the reward function for the tutorial.
    It specifically solves the "Momentum Farming" exploit found in earlier versions.

    **V18 Logic**:
    1. **Anti-Momentum Farming**: Sonic only gets speed rewards when he is near his
       personal 'frontier' (his max_x record). This prevents him from running
       back and forth in a safe area to "mine" reward points.
    2. **Backtrack Credit**: If Sonic walks backwards in specific ramp zones, he
       earns "credit." This mimics the human idea of "needing a run-up" for a jump.
    3. **Universal Victory**: Uses the `level_end_bonus` RAM signal for a clean win bonus.
    """

    def __init__(self, env):
        super().__init__(env)
        self.visited_tiles = set()  # Exploration tracking (16x16 pixel grid)
        self.prev_lives = 3
        self.prev_rings = 0
        self.max_x = 0
        self.prev_x = 0
        self.backtrack_credit = 0.0  # "Gift" to spend on forward speed if we run-up
        self.min_y = None  # Tracks highest altitude reached
        self.stagnant_steps = 0  # Counter for inactivity

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.visited_tiles = set()
        self.prev_lives = info.get("lives", 3)
        self.prev_rings = info.get("rings", 0)
        self.max_x = info.get("x", 0)
        self.prev_x = self.max_x
        self.backtrack_credit = 0.0
        self.min_y = info.get("y", None)
        self.stagnant_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Pull level specific variables from the game RAM data provided by Retro
        curr_x = info.get("x", 0)
        curr_y = info.get("y", 0)
        curr_rings = info.get("rings", 0)
        curr_lives = info.get("lives", 3)
        velocity_x = curr_x - self.prev_x

        # 1. DISCOVERY (Exploration)
        # We give a tiny reward for exploring new 16x16 patches of the world.
        # This prevents the AI from getting "paralyzed" by fear in new areas.
        discovery_bonus = 0.0
        tile = (curr_x // 16, curr_y // 16)
        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            discovery_bonus = 0.02

        # 2. SPIN DASH (Intent)
        # Charging a spin dash (DOWN+B) when stationary.
        # This is CRITICAL for loops. Without this, the AI might never "discover"
        # that it needs to crouch and jump to build speed.
        spin_dash_reward = 0.0
        if action in [7, 9] and abs(velocity_x) < 1:
            spin_dash_reward = 0.1

        # 3. BACKTRACK & MOMENTUM (Physics)
        # The concept of a "Run-up":
        # If the AI walks backwards in a ramp zone, we give it "credit."
        # It can then "spend" that credit for a multiplier when it runs forward.
        in_ramp_zone = (2600 < curr_x < 2800) or (4000 < curr_x < 4400)
        if in_ramp_zone and velocity_x < -1:
            self.backtrack_credit = min(20.0, self.backtrack_credit + 0.5)
        elif velocity_x > 0:
            # Spend credit as we move forward
            self.backtrack_credit = max(0.0, self.backtrack_credit - 0.1)

        # 4. MOMENTUM (High Performance Reward)
        # We want Sonic to go FAST. We use speed squared (speed^2) so high speed
        # is worth MUCH more than low speed.

        # THE "ANTI-FARMING" FIX (V18):
        # Only reward speed when near the 'frontier' (within 300 pixels of his best).
        # Otherwise, the agent might run back and forth in the first level to
        # get infinite points without ever trying to finish the level.
        on_frontier = curr_x > (self.max_x - 300)
        speed = max(0, velocity_x) if on_frontier else 0
        speed_factor = 1.0 + (self.backtrack_credit * 0.1)
        momentum_reward = (speed**2) * 0.02 * speed_factor

        # 5. HORIZONTAL PROGRESS (The Main Goal)
        # The primary reward comes from increasing the 'max_x' record.
        progress_mult = (
            2.0 if curr_x > 2400 else 1.0
        )  # High stakes after the first loop
        progress_reward = 0.0
        if curr_x > self.max_x:
            # We reward the difference: How many new pixels did we reach?
            # Stronger incentive to break personal records
            progress_reward = (
                (curr_x - self.max_x)
                * (1.0 + self.backtrack_credit * 0.5)
                * progress_mult
            )
            self.max_x = curr_x
            # As max_x increases, we "consume" the backtrack credit faster.
            self.backtrack_credit = max(0, self.backtrack_credit - 2.0)
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        # 5. ALTITUDE (Restore V14 Power)
        # We reward climbing platforms (Lower Y = higher up).
        # Threshold lowered to 0.5 to allow rewards while grinding up steep slopes.
        altitude_reward = 0.0
        is_jumping = action in [2, 3, 8, 9]
        can_reward_height = (not is_jumping) or (velocity_x > 0.5)

        if curr_x > 2000 and self.min_y is not None and can_reward_height:
            if curr_y < self.min_y:
                altitude_reward = (self.min_y - curr_y) * 2.0  # Restored to 2.0
                self.min_y = curr_y
        elif self.min_y is None or (curr_x < self.prev_x - 5):
            self.min_y = curr_y

        # 6. STAGNATION RECOVERY
        # If the AI is pushed against a wall for 15 steps (V17 tighter threshold),
        # reward Dash/Jump actions to force environmental exploration.
        recovery_bonus = 0.0
        if self.stagnant_steps > 15 and action in [1, 3, 5] and velocity_x < 0.1:
            if action in [8, 7, 9]:
                recovery_bonus = 0.2

        # 7. HAZARDS & PENALTIES (V17 Spike Fear)
        # We increase the ring penalty to ensure Sonic clearing hazards entirely.
        life_penalty = -2500.0 if curr_lives < self.prev_lives else 0.0
        ring_penalty = 0.0
        if curr_rings < self.prev_rings and curr_lives == self.prev_lives:
            ring_penalty = -1000.0  # Increased to -1000 for V17

        self.prev_lives = curr_lives
        self.prev_rings = curr_rings

        # 8. VICTORY (Universal Signpost Bonus)
        # Reaching the actual signpost in any level.
        win_bonus = 500.0 if info.get("level_end_bonus", 0) > 0 else 0.0
        if win_bonus > 0:
            terminated = True
            print(
                f"--- LEVEL CLEAR DETECTED (Bonus: {info.get('level_end_bonus')}) ---"
            )

        self.prev_x = curr_x

        # Combine everything and scale down to stable range (usually +/- 1.0 per step)
        total_custom = (
            progress_reward
            + spin_dash_reward
            + momentum_reward
            + altitude_reward
            + discovery_bonus
            + life_penalty
            + ring_penalty
            + win_bonus
            + recovery_bonus
        )

        return obs, float(total_custom * 0.01), terminated, truncated, info


class TimeLimitWrapper(gym.Wrapper):
    """
    #### Wrapper: TimeLimitWrapper

    Stops the episode after a fixed number of steps (3 minutes).
    Forces the AI to be efficient. Without this, it might decide that
    the safest way to live is to stand still forever.
    """

    def __init__(self, env, max_steps=10800):  # 10,800 steps = 3 minutes at 60 FPS
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
    #### Wrapper: StagnationWrapper

    Stops the episode if the agent doesn't beat its personal 'max_x' record
    for a long period (30 seconds). This efficiently cuts short "boring"
    episodes where the agent is just running into a corner.
    """

    def __init__(self, env, max_stagnant_steps=1800):  # 30 seconds at 60 FPS
        super().__init__(env)
        self.max_stagnant_steps = max_stagnant_steps
        self.current_stagnant_steps = 0
        self.last_x = 0
        self.max_x = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_x = info.get("x", 0)
        self.last_x = curr_x
        self.current_stagnant_steps += 1

        if self.current_stagnant_steps >= self.max_stagnant_steps:
            # If max_x hasn't increased in 30 seconds, Sonic is stuck.
            # This is much stricter than movement tracking and prevents behavioral loops.
            if curr_x <= self.max_x:
                # Sonic is officially "stuck"
                truncated = True
            else:
                # New record! Reset the timer.
                self.max_x = curr_x
                self.current_stagnant_steps = 0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_stagnant_steps = 0
        obs, info = self.env.reset(**kwargs)
        self.last_x = info.get("x", 0)
        self.max_x = self.last_x
        return obs, info
