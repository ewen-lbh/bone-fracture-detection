import random
from pathlib import Path
from typing import *

import cv2
import gym
import numpy as np
from gym import spaces
from nptyping import NDArray
from numpy import array
from rich import print

from detect import brightness_of, contrast_of, detect_edges, grayscale_of


class EdgeDetectionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def pick_from_dataset(self) -> NDArray:
        return self.preprocess(cv2.imread(random.choice(self.dataset)))
    
    @property
    def dataset_size(self) -> int:
        return len(self.dataset)

    def preprocess(self, image: NDArray) -> NDArray:
        """
        Resize the image to the biggest dimensions of the dataset
        """
        width, height = image.shape[:2]
        if width == self.image_dimensions[0] and height == self.image_dimensions[1]:
            return image

        return cv2.resize(image, self.image_dimensions)

    def biggest_dimensions(self, dataset: list[str]) -> Tuple[int, int]:
        biggest_width = biggest_height = 0
        for image in dataset:
            width, height = cv2.imread(image).shape[:2]
            if width > biggest_width:
                biggest_width = width
            if height > biggest_height:
                biggest_height = height
        return biggest_width, biggest_height

    @property
    def done(self) -> bool:
        return self.info["edges"]["brightness"] in range(
            *self.acceptable_brightness_range
        )

    def __init__(
        self,
        render_mode: Union[str, None],
        acceptable_brightness_range: Tuple[int, int],
        dataset: Path,
        max_increment: int = 5,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.dataset = [str(f) for f in dataset.iterdir() if f.is_file()]
        self.thresholds = [100, 100]
        self.acceptable_brightness_range = acceptable_brightness_range
        self.image_dimensions = self.biggest_dimensions(self.dataset)
        self.max_increment = max_increment

        increment_space = lambda: spaces.Discrete(
            max_increment * 2, start=-max_increment
        )
        pixels_space = lambda width, height: spaces.Box(
            low=array([0, 0]), high=array([width, height]), dtype=np.int16
        )

        self.observation_space_shape = 2*self.image_dimensions[0], self.image_dimensions[1]
        self.observation_space = pixels_space(*self.observation_space_shape)

        self.action_space = spaces.Dict(
            {
                "high_threshold": increment_space(),
                "low_threshold": increment_space(),
                "contrast": spaces.Discrete(3, start=1),
                "brightness": increment_space(),
            }
        )
        self.action_space_size = 4 * increment_space().n

        if render_mode == "human":
            import pygame

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.image_dimensions)
            self.clock = pygame.time.Clock()

        # self.renderer = Renderer(render_mode, self._render_frame)

    @property
    def observation(self) -> NDArray:
        return array([self.source, self.edges]).reshape(*self.observation_space_shape)

    def reward(self, brightness: float) -> float:
        low, hi = self.acceptable_brightness_range

        if brightness in self.acceptable_brightness_range:
            return 1

        offset = abs(
            brightness - self.acceptable_brightness_range[0 if brightness < low else 1]
        )
        width = abs(0 - low if brightness < low else 1 - hi)

        return 1 - offset / width

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "source": {
                "brightness": brightness_of(self.source),
                "contrast": contrast_of(self.source),
                "original": self.original_source,
            },
            "edges": {
                "brightness": brightness_of(self.edges),
                "contrast": contrast_of(self.edges),
            },
        }

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        # Pick a random bone radio image from the set
        self.source, self.edges = detect_edges(
            self.pick_from_dataset(), low=seed or 50, high=seed or 50
        )
        self.source = grayscale_of(self.source)
        self.original_source = self.source.copy()

        return (self.observation, self.info) if return_info else self.observation

    def step(self, action: Tuple[str, int]):
        print(f"with {dict(**action)}", end=" ")
        nudge = lambda param_name: np.clip(
            action[param_name], -self.max_increment, self.max_increment
        )

        self.thresholds[0] += action[ "high_threshold"]
        self.thresholds[1] += action[ "low_threshold"]
        self.source = np.clip(
            self.source.astype("int16") * action["contrast"] + action["brightness"], 
            0, 255
        ).astype("uint8")

        _, self.edges = detect_edges(self.source, *self.thresholds)
        edges_brightness = brightness_of(self.edges)

        print(f"edges brightness is {edges_brightness}", end=" => ")

        return (
            self.observation,
            self.reward(edges_brightness),
            self.done,
            self.info,
        )

    def _render_frame(self, mode: str):
        import pygame

        size = self.edges.shape[1::-1]
        cv2_image = np.repeat(self.edges.reshape(size[1], size[0], 1), 3, axis=2)
        surface = pygame.image.frombuffer(cv2_image.flatten(), size, "RGB")
        surface = surface.convert()

        if mode == "human":
            self.window.blit(surface, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
