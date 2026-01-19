import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

from src.cnn import NatureCNN

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization of layers.
    Think of this as "balancing" the weights before the agent starts learning.
    It prevents the internal gradients from exploding or vanishing early on.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """
    The Actor-Critic Agent.
    
    Architecture:
    1. Shared CNN Backbone: High-speed image processing.
    2. Critic Head: Predicts "How good is this situation?"
    3. Actor Head: Predicts "Which button should I press?"
    """
    def __init__(self, envs):
        super().__init__()
        # features_dim is set to 512.
        # This is the shared brain that both the Actor and Critic use.
        self.network = NatureCNN(envs.single_observation_space.shape)
        
        # --- THE CRITIC (Judge) ---
        # It looks at the 512 game features and outputs 1 number.
        # This number is the "State Value" - the expected total score from here on.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1.0)
        )
        
        # --- THE ACTOR (Player) ---
        # It looks at the same 512 game features and outputs scores for each action.
        # If we have 7 buttons, it outputs 7 "logits" (raw scores).
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        )

    def get_value(self, x):
        """Used during the 'Advantage' calculation to see how much better a state was than expected."""
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        """
        The main intelligence function.
        It converts the 512 high-level game features into a probability distribution.
        """
        hidden = self.network(x)
        logits = self.actor(hidden)
        
        # Use a Categorical distribution to choose one action from many choices.
        # Similar to picking a choice from a menu based on how much you like each item.
        probs = Categorical(logits=logits)
        
        if action is None:
            # During rollout, we sample a new action.
            action = probs.sample()
            
        # We return:
        # 1. action: The winner (e.g., "Right")
        # 2. log_prob: How confident were we in this specific action?
        # 3. entropy: How "random" or "uncertain" is the agent overall?
        # 4. value: The Critic's opinion on this state.
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
