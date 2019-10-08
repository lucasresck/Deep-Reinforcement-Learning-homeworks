import pickle
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import gym

os.chdir("../")
import tf_util
import load_policy
os.chdir("Report/")


def main(expert_policy_file, envname, render, max_timesteps=None, num_rollouts=20):

    os.chdir("../")
    print(os.getcwd())
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')
    os.chdir("Report/")

    with tf.Session():
        tf_util.initialize()
        
        os.chdir("../")
        with open(os.path.join('behavioral_cloning', envname + '.pkl'), 'rb') as f:
            model = pickle.load(f)
        os.chdir("Report/")
            
        finished = 0
        time = 0
        
        performance = []
        performance_std = []
            
        while time < 100:
            print()
            print("Dagger iteration " + str(time))
            print()
            finished = 1
            time += 1
        
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
                    action = model.predict(obs[None,:])
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action[None,:])
                    # print(obs.shape)
                    totalr += r
                    steps += 1
                    # if render:
                    #     env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                if steps < max_steps:
                    finished = 0                        
                returns.append(totalr)
            
            performance.append(np.mean(returns))
            performance_std.append(np.std(returns))
                
            if finished == 1:
                print("Max steps achieved. Model trained.")
                break
                
            labels = policy_fn(np.array(observations))
            model.fit(np.array(observations), labels, batch_size=128, epochs=300, verbose=0) 
            
    return performance, performance_std
    
