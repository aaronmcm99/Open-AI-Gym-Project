# Reinforcement Learning with OpenAI Gym and Stable Baselines3

This project serves as an introduction to using OpenAI Gym for Reinforcement Learning (RL) with two popular RL algorithms, Proximal Policy Optimization (PPO) and Deep Q-Network (DQN), implemented using the Stable Baselines3 library.

## Brief Description of Algorithms:

- **Proximal Policy Optimization (PPO):**
  - PPO is an on-policy RL algorithm that aims to optimize policy functions. It prevents large policy updates by clipping the ratio of new and old policy probabilities. PPO has become popular due to its stability and ease of use.

- **Deep Q-Network (DQN):**
  - DQN is an off-policy RL algorithm used for value-based reinforcement learning. It learns a Q-function that estimates the expected cumulative reward of taking a certain action in a given state. DQN employs experience replay and target networks to improve learning stability and efficiency.

## Overview of the Code:

### 1. Environment Setup:
#### Description
- This section installs necessary packages for Stable Baselines3, Gym, and virtual display.
- It imports required libraries and modules.
- Initializes the CartPole-v1 environment for further use.
```python
# Packages Installation
!pip install stable-baselines3[extra]
!pip install gym pyvirtualdisplay

# Import Statements
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Environment Creation
env = gym.make('CartPole-v1')
```

