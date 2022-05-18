import random
from numpy import array
import numpy as np
from collections import deque

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

from rl_environment import EdgeDetectionEnv

REPLAY_MEMORY_SIZE = 3_000
MIN_REPLAY_MEMORY_SIZE = 1_000


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
        # The output layer has 5 nodes (one node per action)
        output = Dense(units=dense_shape, activation="linear", name="output")(_)

        # Put it all together:
        self.model = Model(inputs=input_layer, outputs=[output])
        self.model.compile(
            optimizer=Adam(lr=0.001),
            loss={"output": "mse"},
            metrics={"output": "accuracy"},
        )


class EdgeDetectionAgent:
    def __init__(
        self,
        name,
        env: EdgeDetectionEnv,
        conv_list,
        dense_list,
        memory_sample_size,
        discount_rate,
    ) -> None:
        self.env = env
        self.conv_list = conv_list
        self.dense_list = dense_list
        self.memory_sample_size = memory_sample_size
        self.discount_rate = discount_rate
        self.name = (
            f"{name}: "
            + ", ".join(f"conv {c}" for c in conv_list)
            + ", ".join(f"dense {d}" for d in dense_list)
        )

        print(env.observation_space_shape)

        self.model = NeuralNetwork(
            conv_list,
            dense_list,
            input_shape=env.observation_space_shape,
            dense_shape=env.action_space_size,
        ).model
        self.target_model = NeuralNetwork(
            conv_list,
            dense_list,
            input_shape=env.observation_space_shape,
            dense_shape=env.action_space_size,
        ).model
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.current_step_count = 0

    def remember(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        fourth = lambda t: t[3]
        first = lambda t: t[0]

        memory_sample = random.sample(self.replay_memory, self.memory_sample_size)
        current_states = array(list(map(first, memory_sample)))
        new_states = array(list(map(fourth, memory_sample)))
        current_q_values = self.model.predict(
            current_states.reshape(-1, *self.env.observation_space_shape)
        )
        future_q_values = self.target_model.predict(
            new_states.reshape(-1, *self.env.observation_space_shape)
        )

        training_states = []
        training_q_values = []

        for index, (starting_state, action, reward, ending_state, done) in enumerate(
            memory_sample
        ):
            new_q_value = (
                reward + self.discount_rate * np.max(future_q_values[index])
                if not done
                else reward
            )

            starting_state_q_values = current_q_values[index]
            starting_state_q_values[action] = new_q_value

            training_states.append(starting_state)
            training_q_values.append(starting_state_q_values)

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
