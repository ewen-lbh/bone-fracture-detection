import gym
from gym import spaces
from numpy import array 
import numpy as np
from typing import *
from gymp.utils.renderer import Renderer

class EdgeDetectionEnv(gym.Env):
    metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 4
    }

    def __init__(self, render_mode: str | None, image_dimensions: Tuple[int, int], max_increment: int = 50):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        increment_space = lambda: spaces.Discrete(max_increment*2, start=-max_increment)
        pixels_space = lambda width, height: spaces.Box(low=array([0, 0]), high=array([width, height]), dtype=np.float32)

        self.observiation_space = spaces.Dict({
            "source": pixels_space(*image_dimensions),
            "edges": pixels_space(*image_dimensions),
        })

        self.action_space = spaces.Dict({
            "high_threshold": increment_space(),
            "low_treshold": increment_space(),
            "contrast": increment_space(),
            "luminosty": increment_space(),
        })

        if render_mode == "human":
            import pygame
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(image_dimensions)
            self.clock = pygame.time.Clock()

        self.renderer = Renderer(render_mode, self._render_frame)


    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        # Pick a random bone radio image from the set
        self.source = 
