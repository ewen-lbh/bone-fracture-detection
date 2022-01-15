"""
Attempt to find the best edge detection (Canny) thresholds for a given image
with a reinforcement learning algorithm.

Following the approach of:
https://keras.io/examples/rl/actor_critic_cartpole/
"""

from pathlib import Path
import numpy as np
from typing import Any

from tensorflow.math import log
from tensorflow import expand_dims, convert_to_tensor
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import layers, Input, optimizers, callbacks
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import GradientTape
from detect import detect_edges, brightness_of
from nptyping import NDArray
import cv2
from rich import print

images_height = 256
images_width = 256
past_rewards_discount_rate = 0.99
max_steps_per_episode = 1000
training_dataset: list[NDArray[Any, Any, 3]] = list(
    map(
        lambda filename: cv2.resize(
            cv2.imread(str(filename)), (images_height, images_width)
        ),
        Path("datasets/radiopaedia").glob("*.png"),
    )
)


class ThresholdsEnvironment:
    """
    Available actions:
    - 1: Increase lower threshold by ε
    - 2: Decrease lower threshold by ε
    - 3: Increase upper threshold by ε
    - 4: Decrease upper threshold by ε
    - 5: Toggle contrast boost

    ⚠ reset needs to be called before the first step runs
    """

    def __init__(self, images: list[NDArray[Any, Any, 3]], ε: int = 1):
        self.lower_threshold = 0
        self.upper_threshold = 1
        self.boost_contrast = False
        self.ε = ε
        self.state = None
        self.images = images
        self.current_image = None

    def next_image(self):
        self.current_image = self.images.pop(0)
        print("Changing image")

    def reset(self):
        self.lower_threshold = 0
        self.upper_threshold = 1
        self.next_image()
        _, edges = detect_edges(
            self.current_image,
            low=self.lower_threshold,
            high=self.upper_threshold,
            boost_contrast=self.boost_contrast,
        )
        return edges

    def step(self, action) -> tuple[NDArray[Any, Any], float, bool]:
        """
        Returns:
        - reward: float
        - done: bool
        """
        # print(
            # f"Executing action [{action}]",
            # {
                # 1: "Increase lower threshold by ε",
                # 2: "Decrease lower threshold by ε",
                # 3: "Increase upper threshold by ε",
                # 4: "Decrease upper threshold by ε",
                # 5: "Toggle contrast boost",
            # }.get(action, "<Unknown>"),
        # )
        if action == 1:
            self.lower_threshold += self.ε
        elif action == 2:
            self.lower_threshold -= self.ε
        elif action == 3:
            self.upper_threshold += self.ε
        elif action == 4:
            self.upper_threshold -= self.ε
        elif action == 5:
            self.boost_contrast = not self.boost_contrast
        else:
            raise ValueError("Invalid action")

        # Reward is 1 - relative difference between actual brightness and 15
        _, edges = detect_edges(
            self.current_image,
            self.lower_threshold,
            self.upper_threshold,
            boost_contrast=self.boost_contrast,
        )
        brightness = brightness_of(edges)

        if abs(brightness - 15) < 2:
            return edges, 0, True

        return edges, 1 - abs(brightness - 15) / 15, False


inputs = layers.Input(shape=(images_height, images_width))
common = layers.Dense(128, activation="relu")(inputs)
action = layers.Dense(
    ("⚠ This needs to be kept up-to-date with the number of actions", 5)[1],
    activation="softmax",
)(common)
critic = layers.Dense(1)(common)

model = Model(inputs=inputs, outputs=[action, critic])
env = ThresholdsEnvironment(training_dataset)

# Training

optimizer = Adam(learning_rate=0.01)
huber_loss = Huber()

action_probs_history = []
critic_value_history = []

rewards_history = []
running_reward = 0
episode_count = 0

while True:
    state = env.reset()
    episode_reward = 0

    with GradientTape() as tape:
        for _ in range(1, max_steps_per_episode):

            state = convert_to_tensor(state)
            state = expand_dims(state, 0)
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            action = np.random.choice(5, p=np.squeeze(action_probs)[0]) + 1
            action_probs_history.append(log(action_probs[0, action]))

            state, reward, done = env.step(action)
            rewards_history.append(reward)

            episode_reward += reward

            if done:
                break

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        returns = []
        discounted_sum = 0
        for reward in reversed(rewards_history):
            discounted_sum = reward + past_rewards_discount_rate * discounted_sum
            returns.append(discounted_sum)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8).tolist()

        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            actor_losses.append(-log_prob * (ret - value))
            critic_losses.append(huber_loss(expand_dims(value, 0), expand_dims(ret, 0)))

        loss_value = sum(actor_losses) + sum(critic_losses)
        optimizer.apply_gradients(
            zip(
                tape.gradient(loss_value, model.trainable_variables),
                model.trainable_variables,
            )
        )

        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    episode_count += 1
    if episode_count % 10 == 0:
        print(f"Episode: {episode_count}")
        print(f"Reward: {running_reward}")
        print(f"Loss: {loss_value}")
        print()

    if running_reward > 200:
        print(f"Solved in {episode_count} episodes")
        break
