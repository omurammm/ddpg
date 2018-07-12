# coding:utf-8


import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras import regularizers
import time

KERAS_BACKEND = 'tensorflow'

ENV_NAME = 'Pendulum-v0'  # Environment name
num_episodes = 12000
initial_replay_size = 20000
batch_size = 64
replay_memory_size = 200000
act_interval = 1
train_interval = 1



epsilon_init = 1.0
epsilon_fin = 0.1
exploration_steps = 100000
gamma = 0.99

tau = 0.001

TRAIN = True

class Agent():
    def __init__(self, env):
        self.env = env
        self.dim_obs = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.epsilon = epsilon_init
        self.epsilon_step = (epsilon_init - epsilon_fin) / exploration_steps
        self.t = 0
        self.repeated_action = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_max_q = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.start = 0

        # Create replay memory
        self.replay_memory = deque()

        self.s = tf.placeholder(tf.float32, [None, self.dim_obs])
        #self.action_input = tf.placeholder(tf.float32, [None, self.num_actions])

        self.st = tf.placeholder(tf.float32, [None, self.dim_obs])

        self.action, policy_network = self.build_policy_network(self.s)
        policy_network_weights = policy_network.trainable_weights

        self.t_action, policy_target_network = self.build_policy_network(self.st)
        policy_target_weights = policy_target_network.trainable_weights



        # Create q network
        self.q_value, q_network = self.build_q_network(self.s, self.action)
        q_network_weights = q_network.trainable_weights

        self.target_q_value, target_network = self.build_q_network(self.st, self.t_action)
        q_target_weights = target_network.trainable_weights


        # Define target network update operation
        self.initialize_q_target = [q_target_weights[i].assign(q_network_weights[i]) for i in range(len(q_target_weights))]
        self.initialize_policy_target = [policy_target_weights[i].assign(policy_network_weights[i]) for i in range(len(policy_target_weights))]



        self.update_q_target = [q_target_weights[i].assign(tau*q_network_weights[i]+(1-tau)*q_target_weights[i]) for i in range(len(q_target_weights))]
        self.update_policy_target = [policy_target_weights[i].assign(tau*policy_network_weights[i]+(1-tau)*policy_target_weights[i]) for i in range(len(policy_target_weights))]

        # Define loss and gradient update operation
        self.y, self.loss, self.critic_update, self.actor_update = self.build_training_op(q_network_weights, policy_network_weights)

        self.sess = tf.InteractiveSession()


        self.sess.run(tf.global_variables_initializer())


        # Initialize target network
        self.sess.run(self.initialize_q_target)
        self.sess.run(self.initialize_policy_target)

    def build_q_network(self, s, action):
        s_input = Input(shape=(self.dim_obs,))
        dense = Dense(400, activation='relu',kernel_regularizer=regularizers.l2(0.01))(s_input)
        a_input = Input(shape=(self.num_actions,))
        hidden = concatenate([dense, a_input])
        hidden = Dense(300, activation='relu',kernel_regularizer=regularizers.l2(0.01))(hidden)
        q_outqut = Dense(1, activation='linear')(hidden)
        model = Model(inputs=[s_input, a_input], outputs=q_outqut)

        #model = Sequential()
        #model.add(Dense(32, activation='relu', input_dim=self.dim_obs))
        #model.add(Dense(32, activation='relu'))
        #model.add(Dense(self.num_actions, activation='linear'))


        q_value = model([s, action])

        return q_value, model

    def build_policy_network(self, s):
        model = Sequential()
        model.add(Dense(400, activation='relu', input_dim=self.dim_obs))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(self.num_actions, activation='tanh'))

        action = model(s)

        return action, model

    def build_training_op(self, q_network_weights, policy_network_weights):
        #a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        #a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        #shape = (BATCH_SIZE,)
        #q_value = self.q_value

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - self.q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        critic_optimizer = tf.train.AdamOptimizer(0.001)
        critic_update = critic_optimizer.minimize(loss, var_list=q_network_weights)

        q_value = tf.reduce_mean(self.q_value)

        actor_optimizer = tf.train.AdamOptimizer(0.0001)
        actor_update = actor_optimizer.minimize(q_value, var_list=policy_network_weights)

        return y, loss, critic_update, actor_update


    def get_action(self, s):
        action = self.repeated_action
        if self.t % act_interval == 0:
            if self.epsilon >= random.random() or self.t < initial_replay_size:
                action = [random.uniform(-2,2)]
            else:
                action = self.action.eval(feed_dict={self.s: [np.float32(s)]})[0]
            self.repeated_action = action
        return action

    def test_get_action(self, s):
        action = self.repeated_action
        if self.t % act_interval == 0:
            action = self.action.eval(feed_dict={self.s: [np.float32(s)]})[0]
                #np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(s)]}))
            self.repeated_action = action
        return action

    def run(self, s, action, R, terminal, s_):

        self.total_reward += R

        R = np.sign(R)

        self.replay_memory.append((s, action, R, s_, terminal))
        if len(self.replay_memory) > replay_memory_size:
            self.replay_memory.popleft()

        if self.t >= initial_replay_size:
            if self.t % train_interval == 0:
                self.train()


            self.sess.run(self.update_q_target)
            self.sess.run(self.update_policy_target)


        self.total_max_q += np.max(self.q_value.eval(feed_dict={self.s: [np.float32(s)]}))
        self.duration += 1
        self.t += 1

        if terminal:
            #Debug
            elapsed = time.time() - self.start
            if self.t < initial_replay_size:
                mode = 'random'
            elif initial_replay_size <= self.t < initial_replay_size + exploration_steps:
                mode = 'explore'
            else:
                mode = 'exploit'

            text = 'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7} / STEP_PER_SECOND: {8:.1f}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_max_q / float(self.duration),
                self.total_loss / (float(self.duration) / float(train_interval)), mode, self.duration/elapsed)
            print(text)

            with open('fx_output.txt','a') as f:
                f.write(text)

            self.total_reward = 0
            self.total_max_q = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        if self.epsilon > epsilon_fin and self.t >= initial_replay_size:
            self.epsilon -= self.epsilon_step

    def train(self):
        s_batch = []
        action_batch = []
        R_batch = []
        next_s_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, batch_size)
        for data in minibatch:
            s_batch.append(data[0])
            action_batch.append(data[1])
            R_batch.append(data[2])
            next_s_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        target_q_value_batch = self.target_q_value.eval(feed_dict={self.st: np.float32(np.array(next_s_batch))})
        y_batch = R_batch + (1 - terminal_batch) * gamma * target_q_value_batch.flatten()
        loss, _ = self.sess.run([self.loss, self.critic_update], feed_dict={
            self.s: np.float32(np.array(s_batch)),
            self.y: y_batch
        })


        _ = self.sess.run(self.actor_update, feed_dict={
            self.s: np.float32(np.array(s_batch)),
        })

        self.total_loss += loss




def main():
    env = gym.make(ENV_NAME)
    agent = Agent(env)
    if TRAIN:
        for _ in range(num_episodes):
            agent.start = time.time()
            terminal = False
            s = env.reset()
            while not terminal:
                action = agent.get_action(s)
                s_, R, terminal, _ = env.step(action)
                env.render()
                agent.run(s, action, R, terminal, s_)
                s = s_


    #for _ in range(10):
    terminal = False
    s = env.reset()

    while not terminal:
        action = agent.test_get_action(s)
        s_, R, terminal = env.step(action)
        env.render()
        agent.run(s, action, R, terminal, s_)
        s = s_



if __name__ == '__main__':
    main()
