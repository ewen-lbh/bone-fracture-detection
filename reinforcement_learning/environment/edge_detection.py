import gym
import cv2
from gym import spaces
from numpy import array 
import numpy as np
from typing import *
from nptyping import NDArray
from gymp.utils.renderer import Renderer
from detect import detect_edges

class EdgeDetectionEnv(gym.Env):
    metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 4
    }

    def pick_from_dataset(self) -> NDArray:
        return self.preprocess(cv2.imread(random.choice(self.dataset)))
    
    def preprocess(self, image: NDArray) -> NDArray:
        """
        Resize the image to the biggest dimensions of the dataset
        """
        width, height = image.shape[:2]
        if width == self.image_dimensions[0] and height == self.image_dimensions[1]:
            return image

        return cv2.resize(image, self.image_dimensions)

    def biggest_dimensions(self, dataset: list[Path]) -> Tuple[int, int]:
        biggest_width = biggest_height = 0
        for image in dataset:
            width, height = image.size
            if width > biggest_width:
                biggest_width = width
            if height > biggest_height:
                biggest_height = height
        return biggest_width, biggest_height

    @property
    def done(self) -> bool:
        return self.info["edges"]["luminosity"] in range(*self.acceptable_brightness_range)

    def __init__(self, render_mode: str | None, max_increment: int = 50, acceptable_brightness_range: Tuple[int, int], dataset: Path):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.dataset = [str(f) for f in dataset.iterdir() if f.is_file()]
        self.acceptable_brightness_range = acceptable_brightness_range
        self.image_dimensions = self.biggest_dimensions(self.dataset)

        increment_space = lambda: spaces.Discrete(max_increment*2, start=-max_increment)
        pixels_space = lambda width, height: spaces.Box(low=array([0, 0]), high=array([width, height]), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "source": pixels_space(*image_dimensions),
            "edges": pixels_space(*image_dimensions),
        })

        self.action_space = spaces.Dict({
            "high_threshold": increment_space(),
            "low_treshold": increment_space(),
            "contrast": increment_space(),
            "brightness": increment_space(),
        })

        if render_mode == "human":
            import pygame
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(image_dimensions)
            self.clock = pygame.time.Clock()

        self.renderer = Renderer(render_mode, self._render_frame)

    @property
    def observation(self) -> Tuple[NDArray, NDArray]:
        return self.source, self.edges

    def reward(self, brightness: float) -> float:
        low, hi = self.acceptable_brightness_range

        if brightness in self.acceptable_brightness_range:
            return 1

        offset = abs(brightness - self.acceptable_brightness_range[0 if brightness < low else 1])
        width = abs(0 - low if brightness < low else 1 - hi)

        return 1 - offset/width

    @property
    def info(self) -> Dict[str, Any]:
        return {
                "source": {
                    "brightness": self.brightness_of(self.source),
                    "contrast": self.contrast_of(self.source),
                    "original": self.original_source,
                },
                "edges": {
                    "brightness": self.brightness_of(self.edges),
                    "contrast": self.contrast_of(self.edges),
                },
        }

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        # Pick a random bone radio image from the set
        self.source, self.edges = detect_edges(self.pick_from_dataset(), low=seed, high=seed)
        self.original_source = self.source.copy()

        return (self.observation, self.info) if return_info else self.observation

    def step(self, action: Tuple[str, int]):
        param, nudge = action
        nudge = np.clip(nudge, -self.action_space.high[param], self.action_space.high[param])

        if param == "high_threshold":
            high += nudge
        elif param == "low_threshold":
            low += nudge
        elif param == "contrast":
            self.source *= nudge
        elif param == "luminosity":
            self.source += nudge

        _, self.edges = detect_edges(self.source, low=low, high=high)

        return self.observation, self.reward(self.brightness_of(self.edges)), self.done, self.info

    def _render_frame(self, mode: str):
        import pygame
        size = self.edges.shape[1::-1]
        cv2_image = np.repeat(self.edges.reshape(size[1], size[0], 1), 3, axis = 2)
        surface = pygame.image.frombuffer(cv2_image.flatten(), size, 'RGB')
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

gym.envs.registration.register(
    id="EdgeDetection-v0",
    entry_point="reinforcement_learning.environment:EdgeDetectionEnv",
    max_episode_steps=1000,
    nondeterministic=True,
)
