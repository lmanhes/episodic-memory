# Episodic Memory

Graph-based memory used by [self-supervised robot pythia](https://github.com/lmanhes/pythia)

## 1. Idea

The memory should be able to replace the classical "episodic buffer" commonly used in reinforcement settings.

Having a graph-based memory allows to store sequences of (action, observation) tuples in such a way that 
it should be possible to use planning algorithm on top of it ([Search on the replay buffer](https://arxiv.org/abs/1906.05253)). It can also be used as a goal-space memory ([Learning latent plans from play](https://learning-from-play.github.io/)). 

One other problem that can resolve a graph-based memory is the storage limit. The classical way of handling that is to remove oldest tuples from the memory. This is a hard limitation, because such a system becomes subject to the catastrophic forgetting problem. Graph-based memory can emulate a "natural decay" wich reinforce useful and importants memories and discard progressively the other ones.

## 2. Features

- [X] Store high-dimensional vectors
- [X] Keep sequences of actions and observations as a multi-directed graph
- [X] Perform fast approximate nearest-neighbors search to find relevant memories
- [X] Implement a natural memory decay
- [X] Random sampling of sequences
- [ ] Planning algorithm

## 3. How it works

## 4. How to use

```shell script
# Install requirements
pip install -r requirements.txt
```

```python
import numpy as np
import random

from memory import EpisodicMemory

max_size = 10000
sim_threshold = 31
vector_dim = 200
stability_start = 1000
actions = ["up", "down", "left", "right"]

memory = EpisodicMemory(base_path='model_files',
                        max_size=max_size,
                        index_sim_threshold=sim_threshold,
                        vector_dim=vector_dim,
                        stability_start=stability_start)

# simulate some actions / perceptions
state_m1 = np.random.random((vector_dim,))
action_m1 = random.choice(actions)
for it in range(30):
    state = np.random.random((vector_dim,))
    memory.update(state_m1, action_m1, state)
    state_m1 = state
    action_m1 = random.choice(actions)
    print(f"states : {memory.n_states}\ttransitions : {memory.n_transitions}\tforgeted states : {memory.forgeted}")

# sample some trajectories
trajectories = memory.tree_memory.sample_trajectories(n=15, horizon=6)
```
