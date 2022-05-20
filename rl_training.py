import random
from datetime import datetime
import numpy as np
from typing import NamedTuple, Tuple, Union
import gym
from rl_environment import EdgeDetectionEnv
from rl_agent import EdgeDetectionAgent
from tqdm import trange
from pathlib import Path
import pygame
from rich import print, traceback

traceback.install()

env = EdgeDetectionEnv(
    render_mode=None,
    acceptable_brightness_range=(7, 15),
    acceptable_segments_count_range=(10, 25),
    dataset=Path("datasets/radiopaedia/cropped"),
    max_thresholds_increment=10,
    max_brightness_increment=3,
)

WINDOW = pygame.display.set_mode((1000, 300))
clock = pygame.time.Clock()


class Params(NamedTuple):
    memory_sample_size: int
    ε_fluctuations: int  # 0 to disable ε fluctuation
    max_episodes_count_without_progress: int = 0  # where progress means an increment in reward. Use 0 to disable ECC
    ε_bounds: Tuple[Union[int, float], Union[int, float]] = (0.001, 1)


def run(env: EdgeDetectionEnv, agent: EdgeDetectionAgent, params: Params):
    episode_reward = 0
    ε = 1
    # ε_values = [ε]
    # rewards_history = []
    # epsiodes_without_progress_count = 0
    ε_decay = params.ε_bounds[1] - (
        params.ε_bounds[1] / int(env.dataset_size / (params.ε_fluctuations or 0.8 * env.dataset_size))
    )

    episode = 0
    while not env.saw_everything:
        reward = 0
        step = 1
        action = 0

        current_state = env.reset()


        while not env.done():
            print(f"{datetime.now():%H:%M:%S}", end=" ")
            if random.random() > ε:
                print(f"[bold][magenta]E[/bold][/magenta] {ε*100:.1f}%", end=" ")
                action = agent.what_do_you_want_to_do(current_state)
            else:
                print(f"[bold][cyan]X[/bold][/cyan] {ε*100:.1f}%", end=" ")
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action, ε)
            print(f"rewarded with {reward}")
            episode_reward += reward

            agent.remember((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

            env.render(WINDOW)

        print("==== episode done! ====")

        if ε > params.ε_bounds[0]:
            print(f"decaying {ε = }")
            ε = max(ε * ε_decay, params.ε_bounds[0])

        if params.ε_fluctuations and episode % int(env.dataset_size / params.ε_fluctuations) == 0:
            print(f"flucuating {ε = }")
            ε = params.ε_bounds[1]

        env.save_settings(agent.name, Path(__file__).parent / "rl_reports")

        print("=======================")
        episode += 1

        # if episode_reward == rewards_history[-1]:
        #     epsiodes_without_progress_count += 1

        # rewards_history.append(episode_reward)

        # if  epsiodes_without_progress_count > params.max_episodes_count_without_progress:
        #     ε = ε_values[-1]


params = Params(memory_sample_size=128, ε_fluctuations=2)
agent = EdgeDetectionAgent(
    "curiosity",
    env,
    conv_list=[32],
    dense_list=[32, 32],
    discount_rate=0.99,
    memory_sample_size=params.memory_sample_size,
    update_target_model_every=5,
)

run(env, agent, params)
