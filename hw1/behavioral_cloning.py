import pickle
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)
        
    data = (expert_data['observations'], expert_data['actions'])
    # print(data[0][0])
    # print(data[1][0])
   
    # print('observations: ' + str(data[0].shape))
    # print('actions: ' + str(data[1].shape))
    model = Sequential()
    model.add(Dense(units=64, input_shape=(len(data[0][0]),), activation='relu'))
    model.add(Dense(units=len(data[1][0])))
    model.compile(loss='mse', optimizer='adam')
    model.fit(data[0], data[1], batch_size=128, epochs=300, verbose=0)
    # print(model.predict(data[0][0][None,:]))
    
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

    with open(os.path.join('behavioral_cloning', args.envname + '.pkl'), 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)   
    
if __name__ == '__main__':
    main()
