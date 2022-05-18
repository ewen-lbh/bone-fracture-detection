import random
import numpy as np
from typing import NamedTuple, Tuple, Union
import gym
from rl_environment import EdgeDetectionEnv
from rl_agent import EdgeDetectionAgent
from tqdm import trange
from pathlib import Path

env = EdgeDetectionEnv(
    render_mode=None,
    acceptable_brightness_range=(8, 15),
    dataset=Path("datasets/radiopaedia/cropped")
)

class Params(NamedTuple):
    memory_sample_size: int
    epsilon_fluctuations: int # 0 to disable epsilon fluctuation
    max_episodes_count_without_progress: int = 0 # where progress means an increment in reward. Use 0 to disable ECC
    epsilon_bounds: Tuple[Union[int, float], Union[int, float]] = (1, 0.001)


def run(env: EdgeDetectionEnv, agent: EdgeDetectionAgent, params: Params):
    episode_reward = 0
    epsilon = 1
    # epsilon_values = [epsilon]
    # rewards_history = []
    # epsiodes_without_progress_count = 0
    epsilon_decay = params.epsilon_bounds[1] - (params.epsilon_bounds[1] / int(env.dataset_size/(params.epsilon_fluctuations or 0.8*env.dataset_size))) 

    for episode in trange(env.dataset_size):
        reward = 0
        step = 1
        action = 0

        current_state = env.reset()

        while not env.done:
            if (rand := random.random()) > epsilon:
                print(f"exploitation ({rand:.3f} > {epsilon})", end=" ")
                action = np.argmax(agent.get_q_values(current_state))
            else:
                print(f"exploration ({rand:.3f} < {epsilon})", end=" ")
                action = env.action_space.sample()
            
            new_state, reward, done, info = env.step(action)
            print(f"rewarded with {reward}")
            episode_reward += reward

            agent.remember((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1
        
        if epsilon > params.epsilon_bounds[0]:
            print(f"decaying ε={epsilon}")
            epsilon = max(epsilon * epsilon_decay, params.epsilon_bounds[0])
        
        if params.epsilon_fluctuations and episode % int(env.dataset_size/params.epsilon_fluctuations) == 0:
            print(f"flucuating ε={epsilon}")
            epsilon = params.epsilon_bounds[1]
        
        # if episode_reward == rewards_history[-1]:
        #     epsiodes_without_progress_count += 1

        # rewards_history.append(episode_reward)

        
        # if  epsiodes_without_progress_count > params.max_episodes_count_without_progress:
        #     epsilon = epsilon_values[-1]
        

params = Params(memory_sample_size=128, epsilon_fluctuations=2)
agent = EdgeDetectionAgent("uranus", env, conv_list=[32], dense_list=[32, 32], discount_rate=0.99, memory_sample_size=params.memory_sample_size)

run(env, agent, params)
