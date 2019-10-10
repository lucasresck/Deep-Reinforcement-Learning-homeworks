import pickle
import os
import tensorflow as tf
import tf_util


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('policy', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    with tf.Session():
        tf_util.initialize()

        print('loading and building expert policy')
        with open(os.path.join(args.policy, args.envname + '.pkl'), 'rb') as f:
            model = pickle.load(f)
        print('loaded and built')

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            steps = 0
            while not done:
                action = model.predict(obs[None, :])
                obs, r, done, _ = env.step(action[None, :])
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break


if __name__ == '__main__':
    main()
