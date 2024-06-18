import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
import cv2
from tqdm.notebook import tqdm
from IPython.display import clear_output
from collections import deque


class Actor():
    def __init__(self, observation_space, action_space, lr=0.001):
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.model = self.build_actor()
        self.target_model = self.build_actor()

    def build_actor(self):
        inputs = keras.Input(shape=(self.observation_space,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(128, activation='relu')(x)
        output = keras.layers.Dense(self.action_space, activation='tanh')(x)

        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr))
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())


class Critic():
    def __init__(self, observation_space, action, lr=0.001):
        self.observation_space = observation_space
        self.action = action
        self.lr = lr
        self.model = self.build_critic('model')
        self.target_model = self.build_critic('target')
        self.target_model.set_weights(self.model.get_weights())
        self.update_target()

    def build_critic(self, name):
        state_input = keras.layers.Input(
            shape=self.observation_space, name='state_input')
        action_input = keras.layers.Input(
            shape=self.action, name='action_input')

        concat = keras.layers.Concatenate()([state_input, action_input])

        hidden1 = keras.layers.Dense(128, activation='relu')(concat)
        hidden2 = keras.layers.Dense(128, activation='relu')(hidden1)
        output = keras.layers.Dense(1, activation='linear')(hidden2)

        model = keras.Model(
            inputs=[state_input, action_input], outputs=output, name=name)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                      loss='mse')
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())


class Agent():
    def __init__(self,
                 observation_space,
                 action_space,
                 buffer_size=10000,
                 gamma=0.99,
                 ):

        self.observation_space = observation_space
        self.action_space = action_space
        self.actor = Actor(observation_space, action_space)
        self.critic = Critic(observation_space, action_space)
        self.buffer = deque(maxlen=buffer_size)
        self.gamma = gamma

    def get_action(self, observation, e=0.1):
        action = self.actor.model(observation)
        if np.random.rand() < e:
            return np.clip(action + np.random.normal(0, 0.1, action.shape), -1, 1)
        return action

    def remember(self, s, a, s_, r):
        self.buffer.append([s, a, s_, r])

    def get_batch(self, batch_size=64):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        terminated = np.array([experience[4] for experience in batch])
        return states, actions, next_states, rewards, terminated

    def train(self, data):
        # if len(self.buffer) < batch_size:
        #     return
        s, a, s_, r = data

        # Train critic
        a_ = self.actor.target_model(s_)
        next_q = self.critic.target_model([s_, a_])
        target_q = r + self.gamma*next_q

        with tf.GradientTape() as tape:
            curr_q = self.critic.model([s, a])
            critic_loss = tf.keras.losses.MSE(target_q, curr_q)

        critic_grad = tape.gradient(
            critic_loss, self.critic.model.trainable_variables)
        
        # print("Critic Gradients:", critic_grad)

        self.critic.model.optimizer.apply_gradients(
            zip(critic_grad, self.critic.model.trainable_variables))

        # Train actor
        with tf.GradientTape() as tape:
            tape.watch(self.actor.model.trainable_variables)
            actions = self.actor.model(s, training=True)
            q_values = self.critic.model([s, actions], training=False) 
            actor_loss = -tf.reduce_mean(q_values)

        actor_grad = tape.gradient(
            actor_loss, self.actor.model.trainable_variables)


        self.actor.model.optimizer.apply_gradients(
            zip(actor_grad, self.actor.model.trainable_variables))

    def update(self):
        self.critic.update_target()
        self.actor.update_target()
