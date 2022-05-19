import random
import json
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
        self.current_image_name = random.choice(self.dataset)
        return self.preprocess(cv2.imread(self.current_image_name))
    
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

        print(f"preprocess: resizing: {width}×{height} -> {self.image_dimensions}")
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

    def done(self) -> bool:
        return int(brightness_of(self.edges)) in range(
            *self.acceptable_brightness_range
        )
    
    def save_settings(self, agent_name: str, into: Path):
        # Used to encode int64 and other numpy number types
        def numpy_encoder(object):
            if isinstance(object, np.generic):
                return object.item()
        assert self.current_image_name is not None
        save_as = into / agent_name / Path(self.current_image_name).stem
        save_as.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{save_as}--info.json").write_text(json.dumps(self.info, default=numpy_encoder))
        cv2.imwrite(f"{save_as}--source.png", self.source)
        cv2.imwrite(f"{save_as}--edges.png", self.edges)
        cv2.imwrite(f"{save_as}--original-source.png", self.original_source)

    def __init__(
        self,
        render_mode: Union[str, None],
        acceptable_brightness_range: Tuple[int, int],
        dataset: Path,
        max_increment: int = 5,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.dataset = [str(f) for f in dataset.iterdir() if f.is_file()]
        self.current_image_name = None
        self.thresholds = [100, 100]
        self.brightness_boost = 0
        self.contrast_multiplier = 1
        self.acceptable_brightness_range = acceptable_brightness_range
        self.image_dimensions = self.biggest_dimensions(self.dataset)
        self.max_increment = max_increment
        self.max_contrast_increment = 3

        increment_space = lambda: spaces.Discrete(
            max_increment * 2, start=-max_increment
        )
        pixels_space = lambda width, height: spaces.Box(
            low=array([0, 0]), high=array([width, height]), dtype=np.int16
        )

        self.observation_space_shape = (2*self.image_dimensions[0], self.image_dimensions[1])
        # self.observation_space_shape gives (2,) instead of (width, height)
        self.observation_space = pixels_space(*self.observation_space_shape)

        self.action_space = spaces.Dict(
            {
                "high_threshold": increment_space(),
                "low_threshold": increment_space(),
                 "contrast": spaces.Discrete(self.max_contrast_increment, start=1),
                 "brightness": increment_space(),
            }
        )
        self.action_space_shape = 2 * max_increment + 2 * max_increment + self.max_contrast_increment + 2 * max_increment

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
        width = abs(0 - low if brightness < low else 255 - hi)

        # print(f"computing reward: {offset} / {width}", end=" ")

        return 1 - (offset / width)

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "source": {
                "brightness": brightness_of(self.source),
                "contrast": contrast_of(self.source),
                # "original": self.original_source,
                "name": self.current_image_name,
            },
            "edges": {
                "brightness": brightness_of(self.edges),
                "contrast": contrast_of(self.edges),
            },
            "settings": {
                "high_threshold": self.thresholds[0],
                "low_threshold": self.thresholds[1],
                "contrast_multiplier": self.contrast_multiplier,
                "brightness_boost": self.brightness_boost,
            }
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

    def step(self, action: OrderedDict):
        print(f"with {dict(**action)}", end=" ")
        nudge = lambda param_name: np.clip(
            action[param_name], -self.max_increment, self.max_increment
        )

        self.thresholds[0] = action["high_threshold"]
        self.thresholds[1] = action["low_threshold"]
        self.contrast_multiplier = 1 + action["contrast"]/10
        self.brightness_boost = action["brightness"]
        self.source = np.clip(
            self.original_source.astype("int16") * self.contrast_multiplier + self.brightness_boost,
            # self.source.astype("int16") + action["brightness"],
            0, 255
        ).astype("uint8")

        # if brightness_of(self.edges) == 0:
        #     print("restarting with original source")
        #     self.source = self.original_source.copy()

        _, self.edges = detect_edges(self.source, *self.thresholds)
        edges_brightness = brightness_of(self.edges)

        print(f"edges brightness is {edges_brightness}", end=" => ")

        return (
            self.observation,
            self.reward(edges_brightness),
            self.done(),
            self.info,
        )

    def render(self, window):
        import pygame

        window.fill((255,255,255))
        self._draw_image(self.edges, window, (0, 0))
        self._draw_image(self.source, window, (self.image_dimensions[0], 0))
        self._draw_image(self.original_source, window, (0, self.image_dimensions[1]))
        self._draw_text(f"edges brightness: {brightness_of(self.edges)} in {self.acceptable_brightness_range} ? {self.done()}", window, (20, self.image_dimensions[1] * 2 + 20))
        pygame.display.update()
    
    def _draw_text(self, text, window, *at):
        import pygame

        pygame.init()
        font = pygame.font.SysFont("monospace", 12)
        text = font.render(text, True, (0,0,0))
        window.blit(text, at)
    
    def _draw_image(self, image, window, *at):
        import pygame
        size = image.shape[1::-1]
        cv2_image = np.repeat(image.reshape(size[1], size[0], 1), 3, axis=2)
        surface = pygame.image.frombuffer(cv2_image.flatten(), size, "RGB")
        surface = surface.convert()
        window.blit(surface, at)

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
