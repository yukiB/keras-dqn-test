from collections import deque
import os

import numpy as np
from keras.layers.core import Dense, Flatten
from keras.layers import Lambda, Input, Convolution2D
from keras.models import model_from_yaml, Model
import keras.callbacks
from keras.optimizers import RMSprop
try:
    from keras.optimizers import RMSpropGraves
except:
    print('You do not have RMSpropGraves')

import keras.backend.tensorflow_backend as KTF
from keras import backend as K
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

losses = {'loss': lambda y_true, y_pred: y_pred,
          'main_output': lambda y_true, y_pred: K.zeros_like(y_pred)}


def loss_func(args):
    import tensorflow as tf
    y_true, y_pred, a = args
    error = tf.abs(y_pred - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    tf.summary.scalar('loss', loss)
    
    return loss


def customized_loss(args):
    import tensorflow as tf
    y_true, y_pred, a = args
    a_one_hot = tf.one_hot(a, K.shape(y_pred)[1], 1.0, 0.0)
    q_value = tf.reduce_sum(tf.mul(y_pred, a_one_hot), reduction_indices=1)
    error = tf.abs(q_value - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    return loss


class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, environment_name, graves=False, ddqn=False):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 5000
        self.learning_rate = 0.00025
        self.discount_factor = 0.9
        self.use_graves = graves
        self.use_ddqn = ddqn
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

        state_input = Input(shape=(1, 16, 16), name='state')
        action_input = Input(shape=[None], name='action', dtype='int32')

        x = Convolution2D(16, 4, 4, border_mode='same', activation='relu', subsample=(2, 2))(state_input)
        x = Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1))(x)
        x = Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        y_pred = Dense(self.n_actions, activation='linear', name='main_output')(x)
        y_true = Input(shape=(3, ), name='y_true')

        loss_out = Lambda(loss_func, output_shape=(1,), name='loss')([y_true, y_pred, action_input])
        self.model = Model(input=[state_input, action_input, y_true], output=[loss_out, y_pred])

        optimizer = RMSprop if not self.use_graves else RMSpropGraves
        self.model.compile(loss=losses,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=['accuracy'])
        self.target_model = copy.copy(self.model)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./log', graph=self.session.graph)


    def init_simple_model(self):

        state_input = Input(shape=(1, 8, 8), name='state')
        action_input = Input(shape=[None], name='action', dtype='int32')
        

        x = Flatten()(state_input)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        y_pred = Dense(3, activation='linear', name='main_output')(x)

        y_true = Input(shape=(3, ), name='y_true')
        loss_out = Lambda(loss_func, output_shape=(1, ), name='loss')([y_true, y_pred, action_input])
        self.model = Model(input=[state_input, action_input, y_true], output=[loss_out, y_pred])
              
        optimizer = RMSprop if not self.use_graves else RMSpropGraves
        self.model.compile(loss=losses,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=['accuracy'])

        self.target_model = copy.copy(self.model)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./log', graph=self.session.graph)


    def update_exploration(self, num):
        if self.exploration > FINAL_EXPLORATION:
            self.exploration -= self.exploration_step * num
            if self.exploration < FINAL_EXPLORATION:
                self.exploration = FINAL_EXPLORATION

    def Q_values(self, states, isTarget=False):
        model = self.target_model if isTarget else self.model
        res = model.predict({'state': np.array([states]),
                             'action': np.array([0]),
                             'y_true': np.array([[0] * self.n_actions])
                             })
        return res[1][0]

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

    def experience_replay(self, step, score=None):
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
                if not self.use_ddqn:
                    v = np.max(self.Q_values(state_j_1, isTarget=True))
                else:
                    v = self.Q_values(state_j_1, isTarget=True)[action_j_index]
                y_j[action_j_index] = reward_j + self.discount_factor * v

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)
            action_minibatch.append(action_j_index)
            
        validation_data = None
        if score != None:
            validation_data = ({'action': np.array(action_minibatch),
                                'state': np.array(state_minibatch),
                                'y_true': np.array(y_minibatch)},
                               [np.zeros([minibatch_size]),
                                np.array(y_minibatch)])
            
        self.model.fit({'action': np.array(action_minibatch),
                        'state': np.array(state_minibatch),
                        'y_true': np.array(y_minibatch)},
                       [np.zeros([minibatch_size]),
                        np.array(y_minibatch)],
                       batch_size=minibatch_size,
                       nb_epoch=1,
                       verbose=0,
                       validation_data=validation_data)

        if self.model.validation_data and hasattr(self, 'summary_op'):
            val_data = self.model.validation_data
            tensors = self.model.inputs
            feed_dict = dict(zip(tensors, val_data))
            result = self.session.run([self.summary_op], feed_dict=feed_dict)
            summary_str = result[0]
            self.summary_writer.add_summary(summary_str, step)
            
        score = self.model.predict({'state': np.array(state_minibatch),
                                    'action': np.array(action_minibatch),
                                    'y_true': np.array(y_minibatch)})
        self.current_loss = score[0][0]

    def load_model(self, model_path=None, simple=False):

        yaml_string = open(os.path.join(f_model,
                                        simple_model_filename if simple else model_filename)).read()
        self.model = model_from_yaml(yaml_string)
        self.model.load_weights(os.path.join(f_model,
                                             simple_weights_filename if simple else weights_filename))

        optimizer = RMSprop if not self.use_graves else RMSpropGraves
        self.model.compile(loss=losses,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=['accuracy'])

    def save_model(self, num=None, simple=False):
        yaml_string = self.model.to_yaml()
        model_name = 'dqn_model{0}.yaml'.format(('_simple' if simple else ''))
        weight_name = 'dqn_model_weights{0}{1}.hdf5'.format(
            (str(num) if num else ''), ('_simple' if simple else ''))
        open(os.path.join(f_model, model_name), 'w').write(yaml_string)
        print('save weights')
        self.model.save_weights(os.path.join(f_model, weight_name))

    def end_session(self):
        KTF.set_session(self.old_session)
