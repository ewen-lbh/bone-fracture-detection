import random
from collections import deque

import numpy as np
from numpy import array
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from rich import print
from rl_environment import EdgeDetectionEnv
from utils import partition

REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 333


class NeuralNetwork:
    # Creates a convolutional block given (filters) number of filters, (dropout) dropout rate,
    # (bn) a boolean variable indecating the use of BatchNormalization,
    # (pool) a boolean variable indecating the use of MaxPooling2D
    def conv_block(self, inp, filters=64, bn=True, pool=True, dropout=0.2):
        _ = Conv2D(filters=filters, kernel_size=3, activation="relu")(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPooling2D(pool_size=(2, 2))(_)
        if dropout > 0:
            _ = Dropout(0.2)(_)
        return _

    def __init__(self, conv_list, dense_list, input_shape, dense_shape):
        print(f"neural: init: conv: {conv_list}; dense: {dense_list}; in: {input_shape}; out: {dense_shape}")
        # Defines the input layer with shape = ENVIRONMENT_SHAPE
        input_layer = Input(shape=(*input_shape, 1))
        print(input_shape, input_layer.shape)
        # Defines the first convolutional block:
        print(f"Constructing block #1")
        _ = self.conv_block(input_layer, filters=conv_list[0], bn=False, pool=False)
        # If number of convolutional layers is 2 or more, use a loop to create them.
        if len(conv_list) > 1:
            for i, c in enumerate(conv_list[1:]):
                print(f"Constructing block #{i + 2}")
                _ = self.conv_block(_, filters=c)
        # Flatten the output of the last convolutional layer.
        _ = Flatten()(_)

        # Creating the dense layers:
        for d in dense_list:
            _ = Dense(units=d, activation="relu")(_)
        # The output layer has 4 nodes (one node per action)
        output = Dense(units=dense_shape, activation="linear", name="output")(_)

        # Put it all together:
        self.model = Model(inputs=input_layer, outputs=[output])
        self.model.compile(
            optimizer=Adam(lr=0.001),
            loss={"output": "mse"},
            metrics={"output": "accuracy"},
        )


class EdgeDetectionAgent:

    ACTION_NAMES = ["high_threshold", "low_threshold", "contrast", "brightness", "blur"]

    def __init__(
        self,
        name,
        env: EdgeDetectionEnv,
        conv_list,
        dense_list,
        memory_sample_size,
        discount_rate,
        update_target_model_every,
    ) -> None:
        self.env = env
        self.conv_list = conv_list
        self.dense_list = dense_list
        self.memory_sample_size = memory_sample_size
        self.discount_rate = discount_rate
        self.update_target_model_every = update_target_model_every
        self.name = f"{name}_conv:{'+'.join(map(str, conv_list))}_dense:{'+'.join(map(str, dense_list))}_mem:{memory_sample_size}_γ:{discount_rate}"

        print(env.observation_space_shape)

        self.model = NeuralNetwork(
            conv_list,
            dense_list,
            input_shape=env.observation_space_shape,
            dense_shape=env.action_space_shape,
        ).model
        self.target_model = NeuralNetwork(
            conv_list,
            dense_list,
            input_shape=env.observation_space_shape,
            dense_shape=env.action_space_shape,
        ).model
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.current_step_count = 0

        print(f"agent {self.name} initialized")

    def remember(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < max(self.memory_sample_size, MIN_REPLAY_MEMORY_SIZE):
            return

        memory_sample = random.sample(self.replay_memory, self.memory_sample_size)
        current_states = array([end for _, _, _, end, _ in memory_sample])
        future_states = array([start for start, _, _, _, _ in memory_sample])
        current_q_values = self.model.predict(current_states.reshape(-1, *self.env.observation_space_shape))
        future_q_values = self.target_model.predict(future_states.reshape(-1, *self.env.observation_space_shape))

        training_states = []
        training_q_values = []

        for index, (current_state, action, reward, future_state, done) in enumerate(memory_sample):
            for action_name, action_idx in self.neural_indices_of(action).items():
                new_q = reward + (
                    self.discount_rate * np.max(self.q_values_of_action(future_q_values[index], action_name))
                    if not done
                    else 0
                )
                try:
                    current_q_values[index, action_idx] = new_q
                except IndexError as e:
                    print(f"Tried indexing {index, action_idx} ({action_name}+={action[action_name]}) in {current_q_values.shape} Q-values")
                    raise e

            training_states.append(current_state)
            training_q_values.append(current_q_values[index])

        self.model.fit(
            x=array(training_states).reshape(-1, *self.env.observation_space_shape),
            y=array(training_q_values),
            batch_size=self.memory_sample_size,
            verbose=0,
            shuffle=False,
            callbacks=[],
        )

        if terminal_state:
            self.current_step_count += 1

        if self.current_step_count % self.update_target_model_every == 0:
            self.target_model.set_weights(self.model.get_weights())

    def get_q_values(self, state):
        return self.model.predict(state.reshape(-1, *self.env.observation_space_shape))

    def what_do_you_want_to_do(self, state):
        q_values = self.get_q_values(state).flatten()
        keys = self.ACTION_NAMES
        sizes = [self.env.action_space_layout[k][0] for k in keys]
        offsets = [self.env.action_space_layout[k][1] for k in keys]
        return {
            keys[i]: np.argmax(q_values_for_key) + offsets[i]
            for i, q_values_for_key in enumerate(partition(q_values, sizes))
        }

    def neural_indices_of(self, action: dict) -> dict[str, int]:
        """
        Returns a map of action names to the index of their values in the neural network's output layer.
        """
        return {name: self.neural_index_of(name, nudge) for name, nudge in action.items()}

    def neural_index_of(self, name: str, nudge: int) -> int:
        """
        Returns the index of the value of the action named `name` in the neural network's output layer.
        """
        cursor = 0
        for action_name in self.ACTION_NAMES:
            size, offset = self.env.action_space_layout[action_name]
            if action_name == name:
                return cursor + (nudge - offset)
            cursor += size

    def q_values_of_action(self, q_values, action_name: str) -> list[float]:
        size, offset = self.env.action_space_layout[action_name]
        max_nudge = size + offset
        start = self.neural_index_of(action_name, 0)
        end = self.neural_index_of(action_name, max_nudge)
        return q_values[start : end + 1]
