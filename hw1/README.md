# CS294-112 HW 1: Imitation Learning

## Setting up

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

## Explanation

This is a homework for imitation learning. There are expert policies, which knows very well each agent.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

The work here is:
- generate data from the experts (that is, observations and actions);
- implement behavioral clonning: fit a model that tries to copy the expert, through its data;
- implement Dagger: when behavioral clonning fails, call expert to tell what's right to do. It's done until the model is "good enough".

## Instructions for running

The commands used bellow are for Linux terminal.

If you wanna see only the results, go to **Results**. If you wanna see only one result (the coolest one), run this:

```python run_agent.py dagger Humanoid-v2 --render --num_rollouts 5```

### Expert data

Run ```demo.bash``` to generate data from the expert policies:

```bash demo.bash```

It uses ```run_expert.py``` to get this data. It will save the expert data at ```expert_data/```.

### Behavioral cloning

You must have expert data. Run ```behavioral_cloning.bash``` to generate a policy for each agent that tries to clone it:

```bash behavioral_cloning.bash```

It uses ```behavioral_cloning.py``` to generate the policies, and the policies are saved at ```behavioral_cloning/```.

### Dagger

You must have behavioral cloning data. Run ```dagger.bash``` to generate a policy for each agent:

```bash dagger.bash```

It uses ```dagger.py``` to generate the policies, and the policies are saved at ```dagger/```.

### Results

You must have Dagger data. Run ```run_agent.bash``` to see the resulting models for each agent:

```bash run_agent.bash```

It uses ```run_agent.py``` to show the resulting models.
