import torch
import numpy as np

class RolloutBuffer:
    """
    A temporary storage for experience collected during interaction with the environment.
    In PPO, we collect a 'rollout' of multiple steps before performing a policy update.
    
    This buffer stores Observations, Actions, Log-Probabilities, Rewards, Done masks, and Values.
    """
    def __init__(self, buffer_size, num_envs, obs_shape, action_shape, device):
        # We use torch.zeros to pre-allocate memory on the GPU for speed.
        # This prevents the program from having to request new memory every single frame.
        self.obs = torch.zeros((buffer_size, num_envs) + obs_shape).to(device)
        self.actions = torch.zeros((buffer_size, num_envs) + action_shape, dtype=torch.long).to(device) # Actions must be Long (integers)
        self.logprobs = torch.zeros((buffer_size, num_envs)).to(device)
        # We store rewards and dones as numpy arrays while collecting data.
        # This is MUCH faster than moving them to the GPU one-by-one every step.
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = torch.zeros((buffer_size, num_envs)).to(device)
        
        self.pos = 0 # Current position in the buffer
        self.buffer_size = buffer_size
        self.device = device

    def add(self, obs, action, logprob, reward, done, value):
        """
        Store a single step of simulation.
        We store tensors directly, but keep rewards/dones as numpy for later.
        """
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.logprobs[self.pos] = logprob
        
        # Store in numpy arrays (fast, stays on CPU)
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value.reshape(-1) # Flatten value array to match batch dimension
        
        self.pos += 1 # Advance to the next slot

    def reset(self):
        """Reset the buffer pointer for the next rollout."""
        self.pos = 0

    def compute_returns_and_advantages(self, last_value, last_done, gamma, gae_lambda):
        # Convert the entire rewards/dones block to GPU at once!
        # This is the "Bulk Transfer" that keeps our speed high.
        rewards = torch.from_numpy(self.rewards).to(self.device)
        dones = torch.from_numpy(self.dones).to(self.device)

        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        
        for t in reversed(range(self.buffer_size)):
            # If this is the last step in the buffer, we use the bootstrap value
            if t == self.buffer_size - 1:
                nextnonterminal = 1.0 - last_done.float()
                nextvalues = last_value
            else:
                # Otherwise, we look at the 'done' flag of the next recorded step
                nextnonterminal = 1.0 - dones[t + 1].float()
                nextvalues = self.values[t + 1]
                
            # TD Error formula: (Current Reward + Discounted Future Value) - Current Predicted Value
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            
            # GAE: Recursive sum of discounted TD errors
            last_gae_lam = delta + gamma * gae_lambda * nextnonterminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + self.values
        return returns, advantages

class PPOAlgo:
    """
    Proximal Policy Optimization (PPO) Clipper Objective.
    This class handles the optimization of the Actor-Critic network.
    """
    def __init__(self, agent, optimizer, device, ent_coef=0.02, vf_coef=0.5, clip_coef=0.2, max_grad_norm=0.5):
        self.agent = agent
        self.optimizer = optimizer
        self.device = device
        self.ent_coef = ent_coef # Weight for exploration (entropy)
        self.vf_coef = vf_coef   # Weight for value function loss
        self.clip_coef = clip_coef # PPO epsilon for clipping ratios
        self.max_grad_norm = max_grad_norm

    def update(self, buffer, minibatch_size=256, epochs=4):
        """
        Perform multiple epochs of PPO updates using the collected buffer.
        """
        # Flatten the buffer: [Steps, Envs, ...] -> [TotalSamples, ...]
        b_obs = buffer.obs.reshape((-1,) + buffer.obs.shape[2:])
        b_logprobs = buffer.logprobs.reshape(-1)
        b_actions = buffer.actions.reshape((-1,) + buffer.actions.shape[2:])
        b_advantages = buffer.advantages.reshape(-1)
        b_returns = buffer.returns.reshape(-1)
        b_values = buffer.values.reshape(-1)

        # Advantage Normalization: Stabilizes training
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        inds = np.arange(len(b_obs))
        loss_info = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": []}
        
        for epoch in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, len(inds), minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]
                
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                # Probability Ratio: pi_new / pi_old
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # KL Divergence check (monitors how much the policy changes)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    loss_info["approx_kl"].append(approx_kl.item())

                mb_advantages = b_advantages[mb_inds]
                
                # Policy loss (The core "Proximal" logic)
                # We want to increase the probability of actions that had positive advantages.
                # However, we CLIP the change to 20% (clip_coef) to prevent the policy from 
                # changing too fast and collapsing.
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # Take the conservative (min benefit) estimate

                # Value loss: How wrong was our Critic?
                # We want the Critic to predict exactly what the 'returns' (future rewards) were.
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() # Squared error

                # Exploration Bonus: We subtract entropy because we want to MAXIMIZE it.
                # Higher entropy means the agent is still trying different things.
                entropy_loss = entropy.mean()
                
                # Total Loss: The final number we tell the computer to minimize.
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                # The actual "Learning" happens here:
                self.optimizer.zero_grad() # Clear old memory
                loss.backward()            # Calculate how to change weights to reduce loss
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm) # Safety: don't let updates be too big
                self.optimizer.step()      # Apply the changes to the neural network
                
                loss_info["pg_loss"].append(pg_loss.item())
                loss_info["v_loss"].append(v_loss.item())
                loss_info["entropy"].append(entropy_loss.item())
                
        return {k: np.mean(v) for k, v in loss_info.items()}
