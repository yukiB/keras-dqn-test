from collections import deque
import os

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import InputLayer, Convolution2D
from keras.models import model_from_yaml
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import copy
from util import clone_model

f_log = './log'
f_model = './models'

model_filename = 'dqn_model.yaml'
weights_filename = 'dqn_model_weights.hdf5'

simple_model_filename = 'dqn_model_simple.yaml'
simple_weights_filename = 'dqn_model_weights_simple.hdf5'

INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
EXPLORATION_STEPS = 500


def loss_func(y_val, y_pred):
    error = tf.abs(y_pred - y_val)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    return loss
        

class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, environment_name):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 5000
        self.learning_rate = 0.000001
        self.discount_factor = 0.9
        self.exploration = INITIAL_EXPLORATION
        self.exploration_step = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION_STEPS
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)

        self.old_session = KTF.get_session()
        self.session = tf.Session('')
        KTF.set_session(self.session)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)
        
        # variables
        self.current_loss = 0.0

    def init_model(self):

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(1, 16, 16)))
        self.model.add(Convolution2D(16, 4, 4, border_mode='same', activation='relu', subsample=(2, 2)))
        self.model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1)))
        self.model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.n_actions, activation='linear'))
        
        self.model.compile(loss=loss_func,
                           optimizer="rmsprop",
                           metrics=['accuracy'])
        self.target_model = copy.copy(self.model)

    def init_simple_model(self):            

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(1, 8, 8)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.n_actions, activation='linear'))
        
        self.model.compile(loss=loss_func,
                           optimizer="rmsprop",
                           metrics=['accuracy'])
        self.target_model = copy.copy(self.model)

    def update_exploration(self, num):
        if self.exploration > FINAL_EXPLORATION:
            self.exploration -= self.exploration_step * num
            if self.exploration < FINAL_EXPLORATION:
                self.exploration = FINAL_EXPLORATION        

    def Q_values(self, states, isTarget=False):
        # Q(state, action) of all actions
        model = self.target_model if isTarget else self.model
        res = model.predict(np.array([states]))

        return res[0]

    def update_target_model(self):
        self.target_model = clone_model(self.model)

    def select_action(self, states, epsilon):
        if np.random.rand() <= epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.Q_values(states))]

    def store_experience(self, states, action, reward, states_1, terminal):
        self.D.append((states, action, reward, states_1, terminal))
        return (len(self.D) >= self.replay_memory_size)

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []
        action_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action') alpha(learing rate) = 1
                v = np.max(self.Q_values(state_j_1, isTarget=True))
                y_j[action_j_index] = reward_j + self.discount_factor * v   # NOQA

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)
            action_minibatch.append(action_j_index)

        # training
        self.model.fit(np.array(state_minibatch), np.array(y_minibatch), verbose=0)

        # for log
        score = self.model.evaluate(np.array(state_minibatch), np.array(y_minibatch), verbose=0)
        self.current_loss = score[0]

    def load_model(self, model_path=None, simple=False):

        yaml_string = open(os.path.join(f_model,
                                        simple_model_filename if simple else model_filename)).read()
        self.model = model_from_yaml(yaml_string)
        self.model.load_weights(os.path.join(f_model,
                                             simple_weights_filename if simple else weights_filename))

        self.model.compile(loss=loss_func,
                           optimizer="rmsprop",
                           metrics=['accuracy'])

    def save_model(self, num=None, simple=False):
        yaml_string = self.model.to_yaml()
        model_name = 'dqn_model{0}{1}.yaml'.format((str(num) if num else ''), ('_simple' if simple else ''))
        weight_name = 'dqn_model_weights{0}{1}.hdf5'.format((str(num) if num else ''), ('_simple' if simple else ''))
        open(os.path.join(f_model, model_name), 'w').write(yaml_string)
        print('save weights')
        self.model.save_weights(os.path.join(f_model, weight_name))

    def end_session(self):
        KTF.set_session(self.old_session)
