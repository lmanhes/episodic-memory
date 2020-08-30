# Episodic Memory

Graph-based memory used by [self-supervised robot pythia](https://github.com/lmanhes/pythia)

## 1. Idea

The memory should be able to replace the classical "episodic buffer" commonly used in reinforcement settings.

Having a graph-based memory allows to store sequences of (action, observation) tuples in such a way that 
it should be possible to use planning algorithm on top of it ([Search on the replay buffer](https://arxiv.org/abs/1906.05253)). It can also be used as a goal-space memory ([Learning latent plans from play](https://learning-from-play.github.io/)). 

One other problem that can resolve a graph-based memory is the storage limit. The classical way of handling that is to remove oldest tuples from the memory. This is a hard limitation, because such a system becomes subject to the catastrophic forgetting problem. Graph-based memory can emulate a "natural decay" wich reinforce useful and importants memories and discard progressively the other ones.

## 2. Features

- Store high-dimensional vectors
- Keep sequences of actions and observations as a multi-directed graph
- Perform fast approximate nearest-neighbors search to find relevant memories
- Implement a natural memory decay
- (TODO) Planning algorithm

## 3. How it works

## 4. How to use

