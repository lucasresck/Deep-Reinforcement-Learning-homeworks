import pickle
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import tf_util
import gym
import load_policy


def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()
    
        with open(os.path.join('behavioral_cloning', args.envname + '.pkl'), 'rb') as f:
            model = pickle.load(f)
            
        finished = 0
        time = 0
            
        while time < 100:
            print()
            print("Dagger iteration " + str(time))
            print()
            finished = 1
            time += 1
        
            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []
            for i in range(args.num_rollouts):
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
                    # if args.render:
                    #     env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                if steps < max_steps:
                    finished = 0                        
                returns.append(totalr)
                
            if finished == 1:
                print("Max steps achieved. Model trained.")
                break
                
            labels = policy_fn(np.array(observations))
            model.fit(np.array(observations), labels, batch_size=128, epochs=300, verbose=0)
            
        with open(os.path.join('dagger', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL) 
        
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
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
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)  
    
if __name__ == '__main__':
    main()
