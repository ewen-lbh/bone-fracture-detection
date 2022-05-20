from datetime import datetime
import random
import inspect
import json
from pathlib import Path
from typing import *
import textwrap

import cv2
import gym
import numpy as np
from gym import spaces
from nptyping import NDArray
from numpy import array
from rich import print
from angles import get_lines_probabilistic, unique_angles

from detect import brightness_of, contrast_of, detect_edges, grayscale_of
from utils import roughly_equals, clip


class EdgeDetectionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def pick_from_dataset(self) -> NDArray:
        self.current_image_name = random.choice(self.dataset)
        print(f"picking {self.current_image_name}")
        return self.preprocess(cv2.imread(self.current_image_name))

    @property
    def saw_everything(self) -> bool:
        return len(self.seen_images) == len(self.dataset)

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
        return int(brightness_of(self.edges)) in range(*self.acceptable_brightness_range) and self.segments_count in range(*self.acceptable_segments_count_range)

    def save_settings(self, agent_name: str, into: Path):
        # Used to encode int64 and other numpy number types
        def numpy_encoder(object):
            if isinstance(object, np.generic):
                return object.item()

        assert self.current_image_name is not None
        save_as = into / agent_name / Path(self.current_image_name).stem
        save_as.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{save_as}--info.json").write_text(json.dumps(self.info, default=numpy_encoder, indent=2))
        cv2.imwrite(f"{save_as}--source.png", self.source)
        cv2.imwrite(f"{save_as}--edges.png", self.edges)
        cv2.imwrite(f"{save_as}--original-source.png", self.original_source)

    @property
    def action_space_shape(self) -> int:
        return sum(v[0] for v in self.action_space_layout.values())

    def __init__(
        self,
        render_mode: Union[str, None],
        acceptable_brightness_range: Tuple[int, int],
        acceptable_segments_count_range: Tuple[int, int],
        dataset: Path,
        max_thresholds_increment: int = 5,
        max_brightness_increment: int = 3,
        max_blur_value: int = 30,
        step_blur_value: int =1,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.dataset = [str(f) for f in dataset.iterdir() if f.is_file()]
        self.seen_images = set()
        self.unique_segment_angles = set()
        self.current_image_name = None
        self.thresholds = [100, 100]
        self.brightness_boost = 0
        self.blur = 0
        self.segments_count = None
        self.contrast_multiplier = 1
        self.acceptable_brightness_range = acceptable_brightness_range
        self.acceptable_segments_count_range = acceptable_segments_count_range
        self.image_dimensions = self.biggest_dimensions(self.dataset)
        self.max_increment = max_thresholds_increment
        self.max_contrast_increment = 1
        self.max_brightness_increment = max_brightness_increment
        self.max_blur_value = max_blur_value // step_blur_value
        self.step_blur_value = step_blur_value
        self.last_winning_edges = array([])
        self.last_winning_thresholds = [None, None]

        pixels_space = lambda width, height: spaces.Box(low=array([0, 0]), high=array([width, height]), dtype=np.int16)

        # we stick the two images horizontally instead of adding a third dimension (2*width, height) instead of (2, width, height)
        self.observation_space_shape = (
            2 * self.image_dimensions[0],
            self.image_dimensions[1],
        )
        # self.observation_space.shape gives (2,) instead of (width, height)
        self.observation_space = pixels_space(*self.observation_space_shape)

        # key: [size, offset]
        self.action_space_layout = {
            "high_threshold": [2 * max_thresholds_increment, -max_thresholds_increment],
            "low_threshold": [2 * max_thresholds_increment, -max_thresholds_increment],
            "contrast": [2*self.max_contrast_increment, -self.max_contrast_increment],
            "brightness": [
                2 * self.max_brightness_increment,
                -self.max_brightness_increment,
            ],
            "blur": [self.max_blur_value, 0],
        }

        self.action_space = spaces.Dict(
            {k: spaces.Discrete(size, start=offset) for k, (size, offset) in self.action_space_layout.items()}
        )

        print(f"Initialzed action space with layout {self.action_space_layout}")

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
        lo, hi = self.acceptable_brightness_range

        if brightness in self.acceptable_brightness_range:
            return 1

        offset = abs(brightness - (lo if brightness < lo else hi))
        width = abs((0 - lo) if brightness < lo else (255 - hi))

        reward = 1 - (offset / width)

        if reward == 1 and self.segments_count is not None:
            lo, hi = self.acceptable_segments_count_range
            offset = abs(self.segments_count - (lo if self.segments_count < lo else hi))
            # en supposant segments count ∈ [0, 10_000[
            width = abs((0 - lo) if self.segments_count < lo else (10_000 - hi))
            return clip(0, 1, 0.25 + (1 - offset / width))
        
        return reward

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "at": f"{datetime.now():%Y-%m-%dT%H:%M:%S}",
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
            "segments": {
                "count": self.segments_count,
                "angles": self.unique_segment_angles,
            },
            "settings": {
                "high_threshold": self.thresholds[0],
                "low_threshold": self.thresholds[1],
                "contrast_multiplier": self.contrast_multiplier,
                "brightness_boost": self.brightness_boost,
                "bilateral_blur_sigmas": self.blur,
            },
        }

    def reset(self, seed=None, return_info=False):
        super().reset(seed=seed)
        # Pick a random bone radio image from the set
        self.source, self.edges = detect_edges(self.pick_from_dataset(), low=seed or 50, high=seed or 50)
        self.source = grayscale_of(self.source)
        self.original_source = self.source.copy()

        return (self.observation, self.info) if return_info else self.observation

    def step(self, action: OrderedDict, ε):
        print(f"with {dict(**action)}", end=" ")

        self.thresholds[0] = clip(20, 150, self.thresholds[0] + action["high_threshold"])
        self.thresholds[1] = clip(20, 150, self.thresholds[1] + action["low_threshold"])
        self.blur = action["blur"]
        self.ε = ε
        self.contrast_multiplier = 1 + clip(0, 5, self.contrast_multiplier*10 - 1 + action["contrast"]) / 10
        self.brightness_boost = clip(0, 30, self.brightness_boost + action["brightness"])
        self.source = np.clip(
            self.original_source.astype("int16") * self.contrast_multiplier + self.brightness_boost,
            # self.source.astype("int16") + action["brightness"],
            0,
            255,
        ).astype("uint8")

        # if brightness_of(self.edges) == 0:
        #     print("restarting with original source")
        #     self.source = self.original_source.copy()

        blurred_source, self.edges = detect_edges(self.source, *self.thresholds, blur=self.blur * self.step_blur_value)
        self.source = grayscale_of(blurred_source)
        edges_brightness = brightness_of(self.edges)

        if roughly_equals(0.1)(edges_brightness, 0, 255):
            print("pullup", end=" ")
            self.source = self.original_source.copy()
            self.contrast_multiplier = 1
            self.brightness_boost = 0
            return (self.observation, -1, False, self.info)

        segments = list(get_lines_probabilistic(self.edges, minimum_length=20))
        self.segments_count = len(segments)
        # 50 mrad ≈ 3°
        self.unique_segment_angles = unique_angles(50e-3, segments)

        print(f"bright {edges_brightness}, #seg {self.segments_count}", end=" => ")

        if (done := self.done()):
            self.last_winning_edges = self.edges.copy()
            self.last_winning_thresholds = self.thresholds.copy()

        return (
            self.observation,
            self.reward(edges_brightness),
            done,
            self.info,
        )

    def render(self, window):
        import pygame

        window.fill((255, 255, 255))
        self._draw_image(self.original_source, window, (0, 0))
        self._draw_text("original", window, 0, self.image_dimensions[1] + 20)
        self._draw_image(self.source, window, (self.image_dimensions[0], 0))
        self._draw_text(f"original * {self.contrast_multiplier} + {self.brightness_boost}\nblur {self.blur * self.step_blur_value}", window, self.image_dimensions[0], self.image_dimensions[1] + 20)
        self._draw_image(self.edges, window, (self.image_dimensions[0]*2, 0))
        self._draw_text(
            f"""
            thresh lo {self.thresholds[0]} hi {self.thresholds[1]}
            bright {brightness_of(self.edges):.2f} 
            segments count {self.segments_count}
            """, window, self.image_dimensions[0]*2, self.image_dimensions[1] + 20)
        self._draw_image(self.last_winning_edges, window, (self.image_dimensions[0]*3, 0))
        self._draw_text(
            f"""
            tresh were lo {self.last_winning_thresholds[0]} hi {self.last_winning_thresholds[1]}
            in {self.acceptable_brightness_range}
            in {self.acceptable_segments_count_range}
            """,
            window,
            self.image_dimensions[0]*3,
            self.image_dimensions[1] + 20,
        )

        self._draw_text(
            f"""
            {self.current_image_name}
            {self.ε*100:.1f}% eXploration {(1-self.ε)*100:.1f}% Exploitation
            """,
            window,
            int(self.image_dimensions[0]*1.5),
            self.image_dimensions[1] + 75,
        )
        pygame.display.update()

    def _draw_text(self, text, window, x, y, width=None):
        import pygame

        text = inspect.cleandoc(text)
        if width is not None:
            text = textwrap.fill(text, width=width)
        pygame.init()
        font = pygame.font.SysFont("monospace", 12)
        for i, line in enumerate(text.splitlines()):
            text_surface = font.render(line, True, (0, 0, 0))
            window.blit(text_surface, (x, y + i * 12))

    def _draw_image(self, image, window, *at):
        import pygame

        if len(image.shape) < 2:
            return 
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
