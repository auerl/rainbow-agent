# rainbow-agent

This is an implementation of the RAINBOW algorithm (Hessel et al. 2017), created
as part of the course work for Udacity's Reinforcement Learning course. The code 
is an extension of Udacities reference DQN implementation where improvements 
described in Hessel et al. (2017) are added.

So far I have added the following RAINBOW extensions:

    1. Double Q-learning
    2. Prioritized Experience Replay
    3. Dueling Networks
    4. Noisy Networks

The code can use OpenAI Gym environments as well as Unity based MLAgent environments.

## To Do

Two more extensions need to be added to make the agent a "full" RAINBOW agent:

    5. Multi-step learning
    6. Distributional RL

Furthermore I am working on adding capabilities to learn tasks directly from pixels.

## Requirements

This is research code and no warranty is given whatsoever. The agent has been tested 
on Ubuntu 18.04, but macOS and Window should be supported as well. 

## Running the code

In the folder `tasks` there are notebooks that apply the agent to solve various
tasks. So far only the Unity Banana Collector environment as the Udacity Project
for which this code is developed, has been solved. Just run the jupyter notebook
with

``` shell
jupyter notebook banana-navigation.ipynb
```

and execute all cells.

