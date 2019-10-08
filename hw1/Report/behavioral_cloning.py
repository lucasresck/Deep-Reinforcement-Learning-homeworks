import pickle
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

os.chdir("../")
import tf_util
os.chdir("Report/")


def main(envname, render, max_timesteps=None, num_rollouts=20):

    with tf.Session():
        tf_util.initialize()

        os.chdir("../")
        with open('behavioral_cloning/' + envname + '.pkl', 'rb') as f:
            model = pickle.load(f)
        os.chdir("Report/")

        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            # print(obs.shape)
            # print(obs)
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = model.predict(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action[None, :])
                # print(obs.shape)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break

            returns.append(totalr)

        return returns, np.mean(returns), np.std(returns)
