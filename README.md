#### RLP: RL Algorithms Implemented by Pytorch
RLP is a set of deep reinforcement learning algorithms which implemented by Pytorch.  
Just want to have a deeper understanding of the idea of these algorithms, and meanwhile provide some useful tools for others.  
If you have some problems, please feel free to discuss.😁  
**Advantages: Easy to understand, Concise, Uniform code format**  
**Notice: this implemented based on OpenAI Spinning Up and Others.**  
***
#### Done (but still need to be optimized)  
Off-Policy:    
* DDPG
* TD3
* SAC  
  
  
On-Policy:   
  
* VPG
* TRPO
* PPO

#### Continue Updating...
* DQN
* ...
***
#### Requirements
* gym
* mujoco-py
* PyTorch(1.0.1)
* Python(3.6)
* mpi4py

#### Run
Eg: For DDPG  
   ` python ddpg.py --env HalfCheetah-v2 ...(other parameters)`

#### Test
Eg: For DDPG  
    `python test_policy.py {ddpg model path} -num {choose a model} ...(other parameters)`

#### Benchmarks

Mujoco
