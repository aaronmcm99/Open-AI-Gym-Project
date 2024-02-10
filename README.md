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
### 2. Initial Testing with Random Actions
#### Description
- This section tests the environment by running a few episodes with random actions.
- Each episode is reset, and random actions are taken until the episode terminates.
- The total score (cumulative reward) for each episode is printed.
```python
# Number of episodes
episodes = 5

for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

env.close()
```
### 3. Training Section
### Description
- Sets up the PPO algorithm with specified parameters.
- Creates a vectorized environment for parallelized training.
- Initializes the PPO model with the Multi-Layer Perceptron (MLP) policy.
- Trains the model for a specified number of time steps.
```python
# PPO Algorithm Setup
env_name = 'CartPole-v1'
vec_env = make_vec_env(env_name, n_envs=4)
model = PPO('MlpPolicy', vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)
```
### 4. Evaluation Section
#### Description
- Evaluates the trained PPO model's performance by running it in the environment for a certain number of evaluation episodes.
- Prints the average reward per episode and standard deviation for reward.
```python
# Evaluate the model
evaluate_policy(model, vec_env, n_eval_episodes=10, render=True)
# Prints(average reward per episode, standard deviation for reward)

vec_env.close()
env.close()
```
### 5. Testing Section
#### Description
- Tests the trained PPO model's performance in the environment by running it for a few episodes.
- The model's action predictions are used to interact with the environment.
- The total score (cumulative reward) for each episode is printed.
```python
# Number of episodes
episodes = 5

for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

env.close()
```
### 6. Callback
#### Description
- Sets up a callback mechanism to stop training when the reward threshold is reached and save the best model during training.
- Monitors training progress and prevents overfitting.
```python
# Callback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(vec_env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path='modelsavepath',
                             verbose=1)

model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log='logpath')
```
### 7. Changing Policies
#### Description
- Modifies the policy network architecture for the PPO algorithm to explore its impact on training and performance.
- Trains the model with the updated policy architecture.
```python
# Changing Policies
net_arch = [dict(pi=[128,128,128,128], vf=[128,128,128,128])]
model = PPO('MlpPolicy', vec_env, verbose=1, policy_kwargs={'net_arch':net_arch})
model.learn(total_timesteps=20000, callback=eval_callback)
vec_env.close()
env.close()
```
### 8. Changing Algoirthm
#### Description
- Switches the algorithm from PPO to DQN to compare the performance of both algorithms on the same environment.
- Initializes the DQN model and trains it for a specified number of time steps.
```python
# Changing Algorithm
from stable_baselines3 import DQN
model= DQN('MlpPolicy', vec_env, verbose=1)
model.learn(total_timesteps=20000)
```

